//! Local voice-session state and elapsed time shared by the plugin surfaces.

use std::{
   collections::HashMap,
   time::{
      Duration,
      Instant,
   },
};

use serde::Deserialize;

#[derive(Clone, Copy)]
pub enum Format {
   Stopwatch,
   Human,
}

impl Format {
   #[must_use]
   pub fn from_name(name: &str) -> Option<Self> {
      match name {
         "stopwatch" => Some(Self::Stopwatch),
         "human" => Some(Self::Human),
         _ => None,
      }
   }

   fn display(self, elapsed: Duration) -> String {
      let total = elapsed.as_secs();
      let days = total / 86_400;
      let hours = total / 3600 % 24;
      let minutes = total / 60 % 60;
      let seconds = total % 60;
      match self {
         Self::Stopwatch if days > 0 => format!("{days}d {hours:02}:{minutes:02}:{seconds:02}"),
         Self::Stopwatch if hours > 0 => format!("{hours:02}:{minutes:02}:{seconds:02}"),
         Self::Stopwatch => format!("{minutes:02}:{seconds:02}"),
         Self::Human if days > 0 => format!("{days}d {hours}h {minutes}m {seconds}s"),
         Self::Human if hours > 0 => format!("{hours}h {minutes}m {seconds}s"),
         Self::Human if minutes > 0 => format!("{minutes}m {seconds}s"),
         Self::Human => format!("{seconds}s"),
      }
   }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Accounts {
   Foreground,
   Concurrent,
}

#[derive(Deserialize)]
struct Ready {
   user:       User,
   session_id: String,
}

#[derive(Deserialize)]
struct User {
   id: String,
}

#[derive(Deserialize)]
pub struct VoiceState {
   #[serde(rename = "user_id")]
   pub user:    String,
   #[serde(rename = "session_id")]
   pub session: String,
   #[serde(rename = "channel_id", deserialize_with = "Deserialize::deserialize")]
   pub channel: Option<String>,
   #[serde(rename = "guild_id")]
   pub guild:   Option<String>,
}

struct Call {
   user:      String,
   channel:   String,
   joined_at: Instant,
}

pub struct CallTimer {
   accounts: Accounts,
   sessions: HashMap<String, String>,
   call:     Option<Call>,
   format:   Option<(String, Format)>,
}

impl Default for CallTimer {
   fn default() -> Self {
      Self::new(Accounts::Foreground)
   }
}

impl CallTimer {
   #[must_use]
   pub fn new(accounts: Accounts) -> Self {
      Self {
         accounts,
         sessions: HashMap::new(),
         call: None,
         format: None,
      }
   }

   /// Tracks READY per local account. Desktop keeps background accounts
   /// connected.
   pub fn ready(&mut self, data: &str) -> bool {
      let Ok(ready) = serde_json::from_str::<Ready>(data) else {
         return false;
      };
      self.note_session(ready.user.id, ready.session_id)
   }

   fn note_session(&mut self, user: String, session: String) -> bool {
      if self.sessions.get(&user) == Some(&session) {
         return false;
      }
      let ended = self
         .call
         .as_ref()
         .is_some_and(|call| self.accounts == Accounts::Foreground || call.user == user);
      if self.accounts == Accounts::Foreground {
         self.sessions.clear();
      }
      self.sessions.insert(user, session);
      if ended {
         self.call = None;
      }
      ended
   }

   pub fn voice_state(&mut self, data: &str, now: Instant) -> bool {
      let Ok(state) = serde_json::from_str::<VoiceState>(data) else {
         return false;
      };
      self.update(&state, now)
   }

   #[must_use]
   pub fn owns(&self, state: &VoiceState) -> bool {
      self.sessions.get(&state.user) == Some(&state.session)
   }

   /// Records a session independently confirmed to be local to this client.
   pub fn confirm(&mut self, state: &VoiceState, now: Instant) -> bool {
      let ended = self.note_session(state.user.clone(), state.session.clone());
      self.update(state, now) || ended
   }

   pub fn update(&mut self, state: &VoiceState, now: Instant) -> bool {
      let Some(session) = self.sessions.get(&state.user) else {
         return false;
      };
      let current = self.call.as_ref().filter(|call| call.user == state.user);
      if &state.session != session {
         // Joining on another device takes over the active call for this
         // account.
         if current.is_some() && state.channel.is_some() {
            self.call = None;
            return true;
         }
         return false;
      }
      if current.map(|call| call.channel.as_str()) == state.channel.as_deref() {
         return false;
      }
      self.call = state.channel.as_ref().map(|channel| {
         Call {
            user:      state.user.clone(),
            channel:   channel.clone(),
            joined_at: now,
         }
      });
      true
   }

   pub fn set_format(&mut self, plugin: &str, format: Format) {
      self.format = Some((plugin.to_owned(), format));
   }

   pub fn clear_format(&mut self, plugin: &str) -> bool {
      if self
         .format
         .as_ref()
         .is_some_and(|(owner, _)| owner == plugin)
      {
         self.format = None;
         true
      } else {
         false
      }
   }

   fn elapsed(&self, now: Instant) -> Option<(Format, Duration)> {
      let (_, format) = self.format.as_ref()?;
      let elapsed = now.saturating_duration_since(self.call.as_ref()?.joined_at);
      Some((*format, elapsed))
   }

   #[must_use]
   pub fn text(&self, now: Instant) -> Option<String> {
      let (format, elapsed) = self.elapsed(now)?;
      Some(format.display(elapsed))
   }

   #[must_use]
   pub fn until_tick(&self, now: Instant) -> Option<Duration> {
      let (_, elapsed) = self.elapsed(now)?;
      let fraction = Duration::from_nanos(u64::from(elapsed.subsec_nanos()));
      Some(Duration::from_secs(1).saturating_sub(fraction))
   }
}

#[cfg(test)]
mod tests {
   use super::*;

   fn ready(timer: &mut CallTimer, user: &str, session: &str) -> bool {
      timer.ready(&serde_json::json!({ "user": { "id": user }, "session_id": session }).to_string())
   }

   fn voice(
      timer: &mut CallTimer,
      user: &str,
      session: &str,
      channel: Option<&str>,
      now: Instant,
   ) -> bool {
      let event =
         serde_json::json!({ "user_id": user, "session_id": session, "channel_id": channel });
      timer.voice_state(&event.to_string(), now)
   }

   #[test]
   fn concurrent_accounts_keep_the_local_call_through_other_accounts_ready() {
      let mut timer = CallTimer::new(Accounts::Concurrent);
      timer.set_format("call-timer", Format::Stopwatch);
      let start = Instant::now();
      assert!(!ready(&mut timer, "1", "desktop"));
      assert!(!voice(&mut timer, "1", "phone", Some("42"), start));
      assert!(voice(&mut timer, "1", "desktop", Some("42"), start));
      assert!(!ready(&mut timer, "2", "background"));
      let now = start + Duration::from_millis(10_200);
      assert!(!voice(&mut timer, "2", "background", None, now));
      assert!(!voice(&mut timer, "1", "desktop", Some("42"), now));
      assert_eq!(timer.text(now).as_deref(), Some("00:10"));
      assert_eq!(timer.until_tick(now), Some(Duration::from_millis(800)));
      assert!(voice(&mut timer, "1", "desktop", Some("43"), now));
      assert_eq!(timer.text(now).as_deref(), Some("00:00"));
      assert!(voice(&mut timer, "1", "phone", Some("43"), now));
      assert_eq!(timer.until_tick(now), None);
   }

   #[test]
   fn foreground_account_changes_and_plugin_unload_clear_the_display() {
      let mut timer = CallTimer::default();
      let start = Instant::now();
      assert!(!ready(&mut timer, "1", "mobile"));
      assert!(voice(&mut timer, "1", "mobile", Some("42"), start));
      timer.set_format("call-timer", Format::Human);
      let now = start + Duration::from_secs(86_403);
      assert_eq!(timer.text(now).as_deref(), Some("1d 0h 0m 3s"));
      assert!(timer.clear_format("call-timer"));
      assert_eq!(timer.text(now), None);
      timer.set_format("call-timer", Format::Stopwatch);
      assert_eq!(timer.text(now).as_deref(), Some("1d 00:00:03"));
      assert!(ready(&mut timer, "2", "other"));
      assert!(!voice(&mut timer, "1", "mobile", Some("42"), now));
      assert_eq!(timer.text(now), None);
   }
}
