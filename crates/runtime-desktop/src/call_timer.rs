//! A non-interactive timer in Desktop Phoenix's native workspace overlay.

use std::{
   sync::{
      Condvar,
      LazyLock,
      Mutex,
      MutexGuard,
      Once,
      PoisonError,
   },
   time::{
      Duration,
      Instant,
   },
};

use phoenix_runtime::{
   call_timer::{
      Accounts,
      CallTimer,
      Format,
      VoiceState,
   },
   gateway,
   store,
};
use rquickjs::{
   Ctx,
   Exception,
   Function,
   Object,
};
use serde::Deserialize;

use crate::{
   gpui::{
      self,
      RawDiv,
      SharedString,
      W_FULL,
      argb_to_hsla,
      gpui_px,
   },
   modal,
};

static TIMER: LazyLock<Mutex<DesktopTimer>> = LazyLock::new(|| {
   Mutex::new(DesktopTimer {
      clock: CallTimer::new(Accounts::Concurrent),
      join:  PendingJoin::default(),
   })
});
static CHANGED: Condvar = Condvar::new();
static WORKER: Once = Once::new();

pub fn supported() -> bool {
   modal::workspace_overlay_ready()
      && gpui::supports_accessories()
      && (gateway::event("VOICE_STATE_UPDATE").is_some()
         || gateway::event("VOICE_STATE_UPDATE_BATCH").is_some())
}

fn timer() -> MutexGuard<'static, DesktopTimer> {
   TIMER.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn observe(event: &str, data: &str) {
   let now = Instant::now();
   let changed = match event {
      "READY" => timer().clock.ready(data),
      "VOICE_STATE_UPDATE" => {
         let Ok(voice) = serde_json::from_str(data) else {
            return;
         };
         observe_voice(voice, now)
      },
      "VOICE_STATE_UPDATE_BATCH" => {
         let Ok(batch) = serde_json::from_str::<VoiceBatch>(data) else {
            return;
         };
         let voices = match batch {
            VoiceBatch::List(voices)
            | VoiceBatch::Named {
               voice_states: voices,
            } => voices,
         };
         let mut changed = false;
         for voice in voices {
            changed |= observe_voice(voice, now);
         }
         changed
      },
      "VOICE_SERVER_UPDATE" => {
         let Ok(update) = serde_json::from_str::<VoiceServer>(data) else {
            return;
         };
         update.endpoint.is_some() && timer().server(update.route, now)
      },
      _ => return,
   };
   if changed {
      refresh();
   }
}

fn observe_voice(voice: VoiceState, now: Instant) -> bool {
   let mut state = timer();
   if state.clock.owns(&voice) {
      return state.clock.update(&voice, now);
   }
   let changed = state.clock.update(&voice, now);
   drop(state);
   let account = foreground_account();
   let confirmed = timer().voice(voice, account.as_deref(), now);
   changed || confirmed
}

fn foreground_account() -> Option<String> {
   let path = store::database(&phoenix_runtime::state()?.files_dir()).ok()?;
   Some(path.parent()?.file_name()?.to_str()?.to_owned())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum VoiceBatch {
   List(Vec<VoiceState>),
   Named { voice_states: Vec<VoiceState> },
}

#[derive(Deserialize)]
struct VoiceServer {
   #[serde(flatten)]
   route:    VoiceRoute,
   endpoint: Option<String>,
}

#[derive(Deserialize)]
struct VoiceRoute {
   #[serde(rename = "guild_id")]
   guild:   Option<String>,
   #[serde(rename = "channel_id")]
   channel: Option<String>,
}

#[derive(Default)]
struct PendingJoin {
   voice:  Option<(VoiceState, Instant)>,
   server: Option<(VoiceRoute, Instant)>,
}

struct DesktopTimer {
   clock: CallTimer,
   join:  PendingJoin,
}

impl DesktopTimer {
   fn voice(&mut self, voice: VoiceState, account: Option<&str>, now: Instant) -> bool {
      if account != Some(voice.user.as_str()) {
         return false;
      }
      self.join.voice = voice.channel.is_some().then_some((voice, now));
      self.confirm_join(now)
   }

   fn server(&mut self, route: VoiceRoute, now: Instant) -> bool {
      self.join.server = Some((route, now));
      self.confirm_join(now)
   }

   fn confirm_join(&mut self, now: Instant) -> bool {
      // Some sessions resume without a READY reducer. A voice-server reply
      // paired with our account's state for the same destination identifies
      // the local client even when this account is signed in on another device.
      const WINDOW: Duration = Duration::from_secs(10);
      let Some(((voice, joined), (route, received))) =
         self.join.voice.as_ref().zip(self.join.server.as_ref())
      else {
         return false;
      };
      if now.saturating_duration_since(*joined.min(received)) > WINDOW {
         return false;
      }
      let matches = route.channel.as_ref().map_or_else(
         || {
            route
               .guild
               .as_ref()
               .is_some_and(|guild| voice.guild.as_ref() == Some(guild))
         },
         |channel| voice.channel.as_ref() == Some(channel),
      );
      if !matches {
         return false;
      }
      let changed = self.clock.confirm(voice, *joined);
      self.join = PendingJoin::default();
      changed
   }
}

pub fn bind<'js>(ctx: &Ctx<'js>, ui: &Object<'js>, plugin: &str) -> rquickjs::Result<()> {
   if !supported() {
      return Ok(());
   }
   let owner = plugin.to_owned();
   ui.set(
      "setCallTimer",
      Function::new(ctx.clone(), move |ctx: Ctx<'js>, value: String| {
         let format = Format::from_name(&value)
            .ok_or_else(|| Exception::throw_type(&ctx, "unknown call timer format"))?;
         set_format(&owner, format);
         Ok::<(), rquickjs::Error>(())
      })?,
   )
}

pub fn set_format(plugin: &str, format: Format) {
   timer().clock.set_format(plugin, format);
   WORKER.call_once(|| {
      if let Err(err) = std::thread::Builder::new()
         .name("sunbird-call-timer".to_owned())
         .spawn(tick)
      {
         log::warn!("call timer: {err}");
      }
   });
   refresh();
}

pub fn clear_format(plugin: &str) {
   if timer().clock.clear_format(plugin) {
      refresh();
   }
}

fn refresh() {
   CHANGED.notify_one();
   modal::force_redraw();
}

#[expect(
   clippy::significant_drop_tightening,
   reason = "the condition variable must release the timer mutex atomically"
)]
fn tick() {
   loop {
      let state = timer();
      let until_tick = state.clock.until_tick(Instant::now());
      if let Some(wait) = until_tick {
         let (state, wake) = CHANGED
            .wait_timeout(state, wait)
            .unwrap_or_else(PoisonError::into_inner);
         drop(state);
         if wake.timed_out() {
            modal::force_redraw();
         }
      } else {
         drop(CHANGED.wait(state).unwrap_or_else(PoisonError::into_inner));
      }
   }
}

pub fn render() -> Option<RawDiv> {
   let text = timer().clock.text(Instant::now())?;
   let mut clock = gpui::str_to_any(&text)?;
   let mut overlay = gpui::raw_div()?;
   let mut pill = gpui::raw_div()?;
   gpui::style_div(&mut overlay, |style| {
      const POSITION_ABSOLUTE: u8 = 1;
      const DISPLAY_FLEX: u8 = 1;
      const ALIGN_FLEX_END: u8 = 3;
      style.set(style.at.position, POSITION_ABSOLUTE);
      style.lengths(&style.at.inset, gpui_px(0.0));
      style.lengths(&style.at.size, W_FULL);
      style.set(style.at.display, DISPLAY_FLEX);
      style.set(style.at.align_items, ALIGN_FLEX_END);
      // Align with the text column after the voice status icon, under the
      // channel subtitle on the desktop sidebar.
      style.length(style.at.padding[2], gpui_px(145.0));
      style.length(style.at.padding[3], gpui_px(62.0));
   });
   gpui::style_div(&mut pill, |style| {
      style.length(style.at.font_size, gpui_px(11.0));
      style.length(style.at.line_height, gpui_px(14.0));
      style.shared_string(style.at.font_family, SharedString::from_str("gg sans mono"));
      style.hsla(style.at.text_color, argb_to_hsla(0xFFB5_BAC1));
   });
   gpui::push_any(&mut pill, &mut clock).then(|| gpui::push_div(overlay, pill))
}

#[cfg(test)]
mod tests {
   use super::*;

   #[test]
   fn desktop_join_can_be_confirmed_without_ready_in_either_event_order() {
      for server_first in [false, true] {
         let mut timer = DesktopTimer {
            clock: CallTimer::new(Accounts::Concurrent),
            join:  PendingJoin::default(),
         };
         timer.clock.set_format("call-timer", Format::Stopwatch);
         let now = Instant::now();
         let voice = || {
            VoiceState {
               user:    "1".to_owned(),
               session: "session".to_owned(),
               channel: Some("2".to_owned()),
               guild:   Some("3".to_owned()),
            }
         };
         let route = VoiceRoute {
            guild:   Some("3".to_owned()),
            channel: None,
         };
         assert!(!timer.voice(voice(), Some("other-account"), now));
         if server_first {
            assert!(!timer.server(route, now));
            assert!(timer.voice(voice(), Some("1"), now));
         } else {
            assert!(!timer.voice(voice(), Some("1"), now));
            assert!(timer.server(route, now));
         }
         assert_eq!(timer.clock.text(now).as_deref(), Some("00:00"));
         assert!(timer.clock.owns(&voice()));
      }
   }
}
