//! Gateway messages read from the host's `@ws` events. Phoenix applies
//! `READY` in the gateway session without passing it through the registry.

use std::sync::OnceLock;

use memchr::memmem;
use phoenix_runtime::gateway;
use phoenix_sigs::registry::EventId;
use phoenix_wire::{
   DispatchEvent,
   decode_dispatch,
   edit_frames,
};
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::{
   bridge::{
      Action,
      Sink,
   },
   host::Event,
   patches::call_timer,
};

/// Discord's compact gateway JSON lets us skip unrelated frames before
/// decoding.
const READY_TAG: &[u8] = br#""t":"READY""#;
const VOICE_STATE_TAG: &[u8] = br#""t":"VOICE_STATE_UPDATE""#;

#[derive(Deserialize)]
struct Message<'a> {
   #[serde(borrow)]
   t: Option<&'a str>,
   #[serde(borrow)]
   d: Option<&'a RawValue>,
}

pub struct Socket;

impl Sink for Socket {
   fn ops(&self, _batch: &[u8]) -> Action {
      Action::Pass
   }

   fn dispatch(&self, frame: &[u8]) -> Action {
      static READY: OnceLock<Option<EventId>> = OnceLock::new();

      if memmem::find(frame, READY_TAG).is_none() && memmem::find(frame, VOICE_STATE_TAG).is_none()
      {
         return Action::Pass;
      }
      let Ok(items) = decode_dispatch(frame) else {
         return Action::Pass;
      };
      // Before the host names load, the body alone tells a gateway message.
      let ws = crate::host::table().and_then(|host| host.event(Event::Ws));
      let edits = items
         .iter()
         .filter_map(|item| {
            let DispatchEvent::Host { event, body, .. } = item.event else {
               return None;
            };
            if ws.is_some_and(|number| event != number) {
               return None;
            }
            let message = serde_json::from_slice::<Message<'_>>(body).ok()?;
            let data = message.d?.get();
            match message.t? {
               "READY" => call_timer::ready(data),
               "VOICE_STATE_UPDATE" => {
                  call_timer::voice_state(data);
                  return None;
               },
               _ => return None,
            }
            let ready = (*READY.get_or_init(|| gateway::event("READY")))
               .filter(|id| gateway::listened(*id))?;
            let json = gateway::intercept(ready, data)?;
            let start = data.as_ptr().addr() - body.as_ptr().addr();
            let head = &frame[item.offset..item.offset + item.len - body.len()];
            let mut out = Vec::with_capacity(head.len() + body.len() - data.len() + json.len());
            out.extend_from_slice(head);
            out.extend_from_slice(&body[..start]);
            out.extend_from_slice(json.as_bytes());
            out.extend_from_slice(&body[start + data.len()..]);
            Some((item.offset, Some(out)))
         })
         .collect::<Vec<_>>();
      if edits.is_empty() {
         return Action::Pass;
      }
      Action::Replace(edit_frames(frame, &edits))
   }
}
