use std::{
   sync::atomic::Ordering,
   time::Instant,
};

use phoenix_runtime::call_timer::Format;
use phoenix_ui_mobile::{
   Op,
   Prop,
   Props,
   ViewKind,
   prop,
};
use phoenix_wire::Value;

use crate::patches::{
   ACTIVE,
   Before,
   Draft,
   Patches,
   Place,
   Slot,
   SlotKind,
   patches,
   worker::{
      Job,
      queue,
   },
};

pub fn set_format(plugin: &str, format: Format) {
   ACTIVE.store(true, Ordering::Release);
   patches().call_timer.set_format(plugin, format);
   queue(Job::RefreshCallTimer);
}

pub fn ready(data: &str) {
   if patches().call_timer.ready(data) {
      queue(Job::RefreshCallTimer);
   }
}

pub fn voice_state(data: &str) {
   if patches().call_timer.voice_state(data, Instant::now()) {
      queue(Job::RefreshCallTimer);
   }
}

impl Patches {
   pub fn refresh_call_timer(&mut self) -> Vec<Op> {
      let root = self.mirror.root();
      let now = Instant::now();
      self.call_timer_tick = self
         .call_timer
         .until_tick(now)
         .and_then(|wait| now.checked_add(wait));
      let text = self.call_timer.text(now);
      let existing = self
         .slots
         .iter()
         .position(|slot| slot.kind == SlotKind::CallTimer);
      let mut ops = Vec::new();
      if let Some(index) = existing {
         let slot = &self.slots[index];
         if root == Some(slot.parent)
            && let Some(text) = &text
            && let Some(&label) = slot.all.first()
         {
            return vec![Op::Update {
               id:    label,
               props: timer_text(text),
            }];
         }
         let slot = self.slots.remove(index);
         if self.mirror.has_kids(slot.parent) {
            ops.push(Op::Children {
               id:  slot.parent,
               ids: self.merged(slot.parent),
            });
         }
         ops.push(Op::Delete { ids: slot.all });
      }
      if let (Some(parent), Some(text)) = (root, text) {
         let mut all = Vec::new();
         let banner = self.emit(timer_banner(&text), None, &mut ops, &mut all);
         self.slots.push(Slot {
            kind: SlotKind::CallTimer,
            anchor: parent,
            parent,
            before: Before::Start,
            place: Place::Screen,
            roots: vec![banner],
            all,
         });
         ops.push(Op::Children {
            id:  parent,
            ids: self.merged(parent),
         });
      }
      ops
   }
}

fn timer_text(text: &str) -> Props {
   vec![
      (Prop::TEXT, Value::Str(text.to_owned())),
      (
         Prop::ACCESSIBILITY_LABEL,
         Value::Str(format!("Call duration {text}")),
      ),
   ]
}

fn timer_banner(text: &str) -> Draft {
   let mut props = timer_text(text);
   props.extend([
      (
         Prop::ACCESSIBILITY_ID,
         Value::Str("phoenix-call-timer".to_owned()),
      ),
      (Prop::COLOR, Value::I64(0xFFF2_F3F5)),
      (Prop::FONT_SIZE, Value::F64(12.0)),
      (Prop::LINE_HEIGHT, Value::F64(16.0)),
      (Prop::FONT_FAMILY, Value::Str("monospace".to_owned())),
   ]);
   let label = Draft {
      kind: ViewKind::Text,
      props,
      kids: Vec::new(),
      press: None,
      src: None,
   };
   let pill = Draft {
      kind:  ViewKind::View,
      props: vec![
         (prop!("background"), Value::I64(0xFF23_2428)),
         (Prop::PADDING, Value::Rect([8.0, 3.0, 8.0, 3.0])),
         (Prop::RADIUS, Value::F64(12.0)),
      ],
      kids:  vec![label],
      press: None,
      src:   None,
   };
   let row = Draft {
      kind:  ViewKind::View,
      props: vec![
         (prop!("width"), Value::Str("fill".to_owned())),
         (prop!("direction"), Value::Str("row".to_owned())),
         (prop!("justify"), Value::Str("center".to_owned())),
      ],
      kids:  vec![pill],
      press: None,
      src:   None,
   };
   Draft {
      kind:  ViewKind::Banner,
      props: vec![
         (prop!("edge"), Value::Str("top".to_owned())),
         (prop!("passthrough"), Value::Bool(true)),
      ],
      kids:  vec![row],
      press: None,
      src:   None,
   }
}
