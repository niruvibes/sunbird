//! Desktop `phoenix.ui` bindings backed by [`phoenix_ui_desktop`] and native
//! GPUI message accessories ([`crate::accessories`]).
//!
//! Native clipboard, toasts, notifications, and `addMessageAccessory` run live
//! on Desktop Phoenix. Remaining mobile-only UI-op slot hooks return harmless
//! no-op removers so cross-platform plugins load cleanly without throwing.

use std::{
   sync::{
      Arc,
      atomic::{
         AtomicU64,
         Ordering,
      },
   },
   time::Duration,
};

use phoenix_runtime::{
   discord::RequestAction,
   js::{
      Engine,
      Registry,
      background::{
         Settlement,
         in_background,
      },
   },
   store::{
      self,
      Cell,
   },
};
use rquickjs::{
   Array,
   CatchResultExt,
   Ctx,
   Exception,
   Function,
   Object,
   Value,
   function::Opt,
};

use crate::{
   accessories::{
      self,
      RenderedAccessory,
   },
   call_timer,
   context_menu::{
      self,
      ActionMeta,
   },
   emoji,
   js::Desktop,
   modal::{
      self,
      ModalButton,
      ModalSection,
      PageModalSpec,
   },
};

#[derive(Clone)]
pub struct DesktopAccessory<'js> {
   pub id:       u64,
   pub plugin:   String,
   pub priority: i32,
   pub render:   Function<'js>,
}

#[derive(Clone)]
pub struct DesktopAction<'js> {
   pub id:       u64,
   pub plugin:   String,
   pub label:    String,
   pub icon:     Option<String>,
   pub danger:   bool,
   pub priority: i32,
   pub visible:  Option<Function<'js>>,
   pub on_press: Function<'js>,
}

impl DesktopAction<'_> {
   pub fn meta(&self) -> ActionMeta {
      ActionMeta {
         id:          self.id,
         label:       self.label.clone(),
         icon:        self.icon.clone(),
         danger:      self.danger,
         priority:    self.priority,
         has_visible: self.visible.is_some(),
      }
   }
}

fn noop_handle<'js>(ctx: &Ctx<'js>) -> rquickjs::Result<Function<'js>> {
   let remove = Function::new(ctx.clone(), || {})?;
   remove.set("update", Function::new(ctx.clone(), || {})?)?;
   remove.set("close", Function::new(ctx.clone(), || {})?)?;
   Ok(remove)
}

fn parse_rendered(value: &Value<'_>, priority: i32) -> Option<RenderedAccessory> {
   if value.is_null() || value.is_undefined() || value.as_bool() == Some(false) {
      return None;
   }
   if let Some(s) = value.as_string() {
      let text = s.to_string().ok()?;
      if text.is_empty() {
         return None;
      }
      return Some(RenderedAccessory {
         priority,
         text,
         color: None,
         tint: None,
      });
   }
   if let Some(obj) = value.as_object() {
      let text = obj
         .get::<_, Option<String>>("text")
         .ok()
         .flatten()
         .unwrap_or_default();
      let color = obj
         .get::<_, Option<f64>>("color")
         .ok()
         .flatten()
         .map(to_argb);
      let tint = obj
         .get::<_, Option<f64>>("tint")
         .ok()
         .flatten()
         .map(to_argb);
      if text.is_empty() && tint.is_none() {
         return None;
      }
      return Some(RenderedAccessory {
         priority,
         text,
         color,
         tint,
      });
   }
   None
}

fn evaluate_accessories_in_ctx<'js>(ctx: &Ctx<'js>, message_id: u64) -> Vec<RenderedAccessory> {
   let Some(registry) = ctx.userdata::<Desktop<'js>>() else {
      return Vec::new();
   };
   let snapshot: Vec<DesktopAccessory<'js>> = registry.accessories.borrow().clone();
   drop(registry);
   if snapshot.is_empty() {
      return Vec::new();
   }

   let id_str = message_id.to_string();
   let Ok(arg) = Object::new(ctx.clone()) else {
      return Vec::new();
   };
   let _ = arg.set("messageId", id_str.as_str());
   let _ = arg.set("channelId", Value::new_null(ctx.clone()));
   let _ = arg.set("currentUserId", Value::new_null(ctx.clone()));
   if let Ok(reply) = Function::new(ctx.clone(), String::new) {
      let _ = arg.set("reply", reply);
   }

   let mut out = Vec::new();
   for acc in snapshot {
      match acc.render.call::<_, Value<'_>>((arg.clone(),)).catch(ctx) {
         Ok(val) if val.is_promise() => {
            if let Some(promise) = val.as_object()
               && let Ok(then) = promise.get::<_, Function<'_>>("then")
            {
               let mid = message_id;
               let prio = acc.priority;
               if let Ok(on_resolve) =
                  Function::new(ctx.clone(), move |ctx: Ctx<'js>, resolved: Value<'js>| {
                     if let Some(item) = parse_rendered(&resolved, prio) {
                        let mut all = evaluate_sync_only(&ctx, mid, Some((prio, item)));
                        all.sort_by_key(|a| a.priority);
                        accessories::set_rendered(mid, all);
                     }
                  })
               {
                  let _ = then.call::<_, Value<'_>>((on_resolve,));
               }
            }
         },
         Ok(val) => {
            if let Some(rendered) = parse_rendered(&val, acc.priority) {
               out.push(rendered);
            }
         },
         Err(err) => {
            log::error!("[{}] addMessageAccessory render: {err}", acc.plugin);
         },
      }
   }
   out
}

fn evaluate_sync_only<'js>(
   ctx: &Ctx<'js>,
   message_id: u64,
   extra: Option<(i32, RenderedAccessory)>,
) -> Vec<RenderedAccessory> {
   let mut out = Vec::new();
   if let Some((_, item)) = extra {
      out.push(item);
   }
   let Some(registry) = ctx.userdata::<Desktop<'js>>() else {
      return out;
   };
   let snapshot: Vec<DesktopAccessory<'js>> = registry.accessories.borrow().clone();
   drop(registry);

   let id_str = message_id.to_string();
   let Ok(arg) = Object::new(ctx.clone()) else {
      return out;
   };
   let _ = arg.set("messageId", id_str.as_str());
   let _ = arg.set("channelId", Value::new_null(ctx.clone()));
   let _ = arg.set("currentUserId", Value::new_null(ctx.clone()));

   for acc in snapshot {
      if let Ok(val) = acc.render.call::<_, Value<'_>>((arg.clone(),)).catch(ctx)
         && !val.is_promise()
         && let Some(rendered) = parse_rendered(&val, acc.priority)
         && !out.contains(&rendered)
      {
         out.push(rendered);
      }
   }
   out
}

/// Evaluates every registered desktop `addMessageAccessory` for `message_id`.
pub fn render_accessories(engine: &Engine, message_id: u64) -> Vec<RenderedAccessory> {
   engine.with(|ctx| evaluate_accessories_in_ctx(&ctx, message_id))
}

#[expect(
   clippy::cast_possible_truncation,
   clippy::cast_sign_loss,
   reason = "ARGB colors are 0..=0xFFFFFFFF, and `as` saturates the rest"
)]
const fn to_argb(color: f64) -> u32 {
   color as u32
}

fn add_message_accessory<'js>(
   ctx: &Ctx<'js>,
   plugin: &str,
   options: &Object<'js>,
) -> rquickjs::Result<Function<'js>> {
   static NEXT_ID: AtomicU64 = AtomicU64::new(1);

   let render = options.get::<_, Function<'js>>("render")?;
   let priority = options.get::<_, Option<i32>>("priority")?.unwrap_or(0);
   let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

   let count = {
      let registry = ctx
         .userdata::<Desktop<'js>>()
         .ok_or_else(|| Exception::throw_internal(ctx, "no registry"))?;
      let mut list = registry.accessories.borrow_mut();
      list.push(DesktopAccessory {
         id,
         plugin: plugin.to_owned(),
         priority,
         render,
      });
      list.len()
   };
   accessories::set_active_count(count);
   accessories::clear_cache();

   let remove = Function::new(ctx.clone(), move |ctx: Ctx<'js>| {
      if let Some(registry) = ctx.userdata::<Desktop<'js>>() {
         let mut list = registry.accessories.borrow_mut();
         list.retain(|entry| entry.id != id);
         accessories::set_active_count(list.len());
         accessories::clear_cache();
         accessories::invalidate_discord_windows();
      }
   })?;
   remove.set(
      "update",
      Function::new(ctx.clone(), |ctx: Ctx<'js>, message_id: Opt<String>| {
         if let Some(mid) = message_id.0.and_then(|s| s.parse::<u64>().ok()) {
            let items = evaluate_accessories_in_ctx(&ctx, mid);
            accessories::set_rendered(mid, items);
         } else {
            accessories::clear_cache();
            accessories::invalidate_discord_windows();
         }
      })?,
   )?;
   Ok(remove)
}

fn cached_message_and_me(channel_id: u64, message_id: u64) -> (Option<String>, Option<String>) {
   let Some(state) = phoenix_runtime::state() else {
      return (None, None);
   };
   let Ok(db) = store::database(&state.files_dir()) else {
      return (None, None);
   };
   // Phoenix keys messages `<channel>:<message>`, each zero-padded to 20 digits.
   let rows = if channel_id != 0 {
      store::query(&db, "SELECT v FROM messages WHERE k = ?", &[format!(
         "{channel_id:020}:{message_id:020}"
      )])
   } else {
      store::query(&db, "SELECT v FROM messages WHERE k LIKE ? LIMIT 1", &[
         format!("%:{message_id:020}"),
      ])
   };
   let msg_json = rows
      .ok()
      .and_then(|rows| rows.first("v").and_then(Cell::text).map(str::to_owned));
   let me_id = store::meta(&db, "me")
      .ok()
      .flatten()
      .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
      .and_then(|val| {
         val.get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
      });
   (msg_json, me_id)
}

fn build_message_context<'js>(
   ctx: &Ctx<'js>,
   message_id: u64,
   channel_id: u64,
   include_message: bool,
) -> rquickjs::Result<Object<'js>> {
   let arg = Object::new(ctx.clone())?;
   let id_str = message_id.to_string();
   arg.set("messageId", id_str.as_str())?;
   if channel_id != 0 {
      let ch_str = channel_id.to_string();
      arg.set("channelId", ch_str.as_str())?;
   } else {
      arg.set("channelId", Value::new_null(ctx.clone()))?;
   }
   let (msg_json, me_id) = if include_message {
      let memory = context_menu::menu_message(message_id);
      let (cached, me) = cached_message_and_me(channel_id, message_id);
      (memory.or(cached), me)
   } else {
      (None, None)
   };
   if let Some(json) = msg_json
      && let Ok(parsed) = ctx.json_parse(json)
   {
      arg.set("message", parsed)?;
   } else {
      arg.set("message", Value::new_null(ctx.clone()))?;
   }
   if let Some(me) = me_id {
      arg.set("currentUserId", me.as_str())?;
   } else {
      arg.set("currentUserId", Value::new_null(ctx.clone()))?;
   }
   if let Ok(reply) = Function::new(ctx.clone(), String::new) {
      arg.set("reply", reply)?;
   }
   Ok(arg)
}

/// Evaluates an `addMessageAction`'s synchronous `visible(ctx)` predicate.
pub fn check_action_visible(
   engine: &Engine,
   action_id: u64,
   message_id: u64,
   channel_id: u64,
) -> bool {
   engine.with(|ctx| {
      let Some((plugin, visible)) = ctx.userdata::<Desktop<'_>>().and_then(|registry| {
         let list = registry.actions.borrow();
         let entry = list.iter().find(|e| e.id == action_id)?;
         Some((entry.plugin.clone(), entry.visible.clone()?))
      }) else {
         return true;
      };
      let Ok(arg) = build_message_context(&ctx, message_id, channel_id, true) else {
         return true;
      };
      match visible.call::<_, Value<'_>>((arg,)).catch(&ctx) {
         Ok(val) if val.is_promise() => true,
         Ok(val) => val.as_bool().unwrap_or(true),
         Err(err) => {
            log::error!("[{plugin}] addMessageAction visible: {err}");
            false
         },
      }
   })
}

/// Invokes a registered `addMessageAction`'s `onPress(ctx)` callback.
pub fn invoke_message_action(engine: &Engine, action_id: u64, message_id: u64, channel_id: u64) {
   engine.with(|ctx| {
      let Some((plugin, on_press)) = ctx.userdata::<Desktop<'_>>().and_then(|registry| {
         let list = registry.actions.borrow();
         let entry = list.iter().find(|e| e.id == action_id)?;
         Some((entry.plugin.clone(), entry.on_press.clone()))
      }) else {
         return;
      };
      let Ok(arg) = build_message_context(&ctx, message_id, channel_id, true) else {
         return;
      };
      if let Err(err) = on_press.call::<_, Value<'_>>((arg,)).catch(&ctx) {
         log::error!("[{plugin}] addMessageAction onPress: {err}");
      }
   });
}

fn add_message_action<'js>(
   ctx: &Ctx<'js>,
   plugin: &str,
   options: &Object<'js>,
) -> rquickjs::Result<Function<'js>> {
   static NEXT_ID: AtomicU64 = AtomicU64::new(1);

   let label = options.get::<_, String>("label")?;
   let icon = options.get::<_, Option<String>>("icon")?;
   let danger = options.get::<_, Option<bool>>("danger")?.unwrap_or(false);
   let priority = options.get::<_, Option<i32>>("priority")?.unwrap_or(0);
   let visible = options.get::<_, Option<Function<'js>>>("visible")?;
   let on_press = options.get::<_, Function<'js>>("onPress")?;
   let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

   let meta = {
      let registry = ctx
         .userdata::<Desktop<'js>>()
         .ok_or_else(|| Exception::throw_internal(ctx, "no registry"))?;
      let mut list = registry.actions.borrow_mut();
      list.push(DesktopAction {
         id,
         plugin: plugin.to_owned(),
         label,
         icon,
         danger,
         priority,
         visible,
         on_press,
      });
      list.iter().map(DesktopAction::meta).collect()
   };
   context_menu::sync_actions(meta);

   let remove = Function::new(ctx.clone(), move |ctx: Ctx<'js>| {
      if let Some(registry) = ctx.userdata::<Desktop<'js>>() {
         let mut list = registry.actions.borrow_mut();
         list.retain(|entry| entry.id != id);
         let meta = list.iter().map(DesktopAction::meta).collect();
         context_menu::sync_actions(meta);
      }
   })?;
   remove.set("update", Function::new(ctx.clone(), || {})?)?;
   Ok(remove)
}

fn invoke_page_button(button_id: u64) {
   let Some(engine) = phoenix_runtime::engine() else {
      return;
   };
   engine.with(|ctx| {
      let Some((plugin, on_press)) = ctx.userdata::<Desktop<'_>>().and_then(|registry| {
         let map = registry.page_buttons.borrow();
         let (_, plugin, func) = map.get(&button_id)?;
         Some((plugin.clone(), func.clone()))
      }) else {
         return;
      };
      if let Err(err) = on_press.call::<_, Value<'_>>(()).catch(&ctx) {
         log::error!("[{plugin}] openPage button onPress: {err}");
      }
   });
}

fn open_page<'js>(
   ctx: &Ctx<'js>,
   plugin: &str,
   options: &Object<'js>,
) -> rquickjs::Result<Object<'js>> {
   static NEXT_PAGE: AtomicU64 = AtomicU64::new(1);
   static NEXT_BTN: AtomicU64 = AtomicU64::new(1);

   let page_id = NEXT_PAGE.fetch_add(1, Ordering::Relaxed);
   let title = options
      .get::<_, Option<String>>("title")?
      .unwrap_or_else(|| "Sunbird".to_owned());
   let sections_js = options.get::<_, Option<Array<'js>>>("sections")?;

   let mut sections = Vec::new();
   let mut new_buttons = Vec::new();
   if let Some(arr) = sections_js {
      for entry in arr.iter::<Object<'js>>() {
         let Ok(sec) = entry else {
            continue;
         };
         let kind = sec.get::<_, Option<String>>("type")?.unwrap_or_default();
         let sec_title = sec.get::<_, Option<String>>("title")?;
         match kind.as_str() {
            "buttons" => {
               let mut buttons = Vec::new();
               if let Some(btns) = sec.get::<_, Option<Array<'js>>>("buttons")? {
                  for item in btns.iter::<Object<'js>>() {
                     let Ok(btn) = item else {
                        continue;
                     };
                     let Ok(label) = btn.get::<_, String>("label") else {
                        continue;
                     };
                     let danger = btn.get::<_, Option<bool>>("danger")?.unwrap_or(false);
                     let on_press = btn.get::<_, Option<Function<'js>>>("onPress")?;
                     let btn_id = NEXT_BTN.fetch_add(1, Ordering::Relaxed);
                     if let Some(func) = on_press {
                        new_buttons.push((btn_id, func));
                     }
                     buttons.push(ModalButton {
                        label,
                        danger,
                        on_press: Arc::new(move || invoke_page_button(btn_id)),
                     });
                  }
               }
               sections.push(ModalSection::Buttons {
                  title: sec_title,
                  buttons,
               });
            },
            "code" => {
               let text = sec.get::<_, Option<String>>("text")?.unwrap_or_default();
               let monospace = sec.get::<_, Option<bool>>("monospace")?.unwrap_or(true);
               let highlight = sec.get::<_, Option<bool>>("highlight")?.unwrap_or(false);
               sections.push(ModalSection::Code {
                  title: sec_title,
                  text,
                  monospace,
                  highlight,
               });
            },
            "text" => {
               let text = sec.get::<_, Option<String>>("text")?.unwrap_or_default();
               sections.push(ModalSection::Text {
                  title: sec_title,
                  text,
               });
            },
            _ => {},
         }
      }
   }

   if let Some(registry) = ctx.userdata::<Desktop<'js>>() {
      let mut map = registry.page_buttons.borrow_mut();
      for (btn_id, func) in new_buttons {
         map.insert(btn_id, (page_id, plugin.to_owned(), func));
      }
   }

   let modal_id = modal::open_page(PageModalSpec { title, sections });
   let obj = Object::new(ctx.clone())?;
   obj.set(
      "close",
      Function::new(ctx.clone(), move |ctx: Ctx<'js>| {
         modal::close_page(modal_id);
         if let Some(registry) = ctx.userdata::<Desktop<'js>>() {
            registry
               .page_buttons
               .borrow_mut()
               .retain(|_, (pid, ..)| *pid != page_id);
         }
      })?,
   )?;
   Ok(obj)
}

/// Wires desktop `phoenix.ui`, `phoenix.messages.onSend` / `onEmojiPick`, and
/// `phoenix.commands` onto the plugin's API objects.
pub fn desktop_patch_api<'js>(
   ctx: &Ctx<'js>,
   plugin: &str,
   api: &Object<'js>,
   messages: &Object<'js>,
) -> rquickjs::Result<()> {
   let name = plugin.to_owned();
   messages.set(
      "onSend",
      Function::new(
         ctx.clone(),
         move |ctx: Ctx<'js>, function: Function<'js>| on_send(&ctx, &name, function),
      )?,
   )?;
   let name = plugin.to_owned();
   messages.set(
      "onEmojiPick",
      Function::new(
         ctx.clone(),
         move |ctx: Ctx<'js>, _function: Opt<Value<'js>>| on_emoji_pick(&ctx, &name),
      )?,
   )?;
   let commands = Object::new(ctx.clone())?;
   commands.set(
      "register",
      Function::new(ctx.clone(), |ctx: Ctx<'js>, _options: Opt<Value<'js>>| {
         noop_handle(&ctx)
      })?,
   )?;
   commands.set(
      "list",
      Function::new(ctx.clone(), |ctx: Ctx<'js>| rquickjs::Array::new(ctx))?,
   )?;
   api.set("commands", commands)?;
   api.set("ui", desktop_ui_object(ctx, plugin)?)
}

pub fn desktop_ui_object<'js>(ctx: &Ctx<'js>, plugin: &str) -> rquickjs::Result<Object<'js>> {
   let ui = Object::new(ctx.clone())?;
   call_timer::bind(ctx, &ui, plugin)?;
   ui.set(
      "toast",
      Function::new(ctx.clone(), |message: String| {
         phoenix_ui_desktop::toast(&message);
      })?,
   )?;
   ui.set(
      "copy",
      Function::new(ctx.clone(), |ctx: Ctx<'js>, text: String| {
         phoenix_ui_desktop::copy(&text)
            .map_err(|err| Exception::throw_message(&ctx, &err.to_string()))
      })?,
   )?;
   ui.set(
      "notify",
      Function::new(ctx.clone(), |ctx: Ctx<'js>, options: Object<'js>| {
         let title = options
            .get::<_, Option<String>>("title")?
            .unwrap_or_default();
         let body = options.get::<_, Option<String>>("body")?;
         phoenix_ui_desktop::notify(&title, body.as_deref());
         noop_handle(&ctx)
      })?,
   )?;

   let owner = plugin.to_owned();
   ui.set(
      "addMessageAccessory",
      Function::new(ctx.clone(), move |ctx: Ctx<'js>, options: Object<'js>| {
         add_message_accessory(&ctx, &owner, &options)
      })?,
   )?;

   let action_owner = plugin.to_owned();
   ui.set(
      "addMessageAction",
      Function::new(ctx.clone(), move |ctx: Ctx<'js>, options: Object<'js>| {
         add_message_action(&ctx, &action_owner, &options)
      })?,
   )?;

   let page_owner = plugin.to_owned();
   ui.set(
      "openPage",
      Function::new(ctx.clone(), move |ctx: Ctx<'js>, options: Object<'js>| {
         open_page(&ctx, &page_owner, &options)
      })?,
   )?;

   // Graceful no-op shims for remaining mobile UI-op surfaces so cross-platform
   // plugins load without throwing on desktop.
   for slot_fn in [
      "addHeaderButton",
      "addComposerButton",
      "addProfileSection",
      "addProfileTab",
      "hide",
      "patchProps",
      "onEvent",
      "onNode",
      "mount",
      "dispatch",
      "update",
   ] {
      ui.set(
         slot_fn,
         Function::new(ctx.clone(), |ctx: Ctx<'js>| noop_handle(&ctx))?,
      )?;
   }
   for query_fn in ["find", "findDown"] {
      ui.set(
         query_fn,
         Function::new(ctx.clone(), |ctx: Ctx<'js>| rquickjs::Array::new(ctx))?,
      )?;
   }
   for nullable_fn in ["node", "root"] {
      ui.set(
         nullable_fn,
         Function::new(ctx.clone(), |ctx: Ctx<'js>| Value::new_null(ctx))?,
      )?;
   }
   ui.set(
      "choose",
      Function::new(ctx.clone(), |ctx: Ctx<'js>, options: Object<'js>| {
         choose_dialog(&ctx, &options)
      })?,
   )?;
   ui.set(
      "showSheet",
      Function::new(ctx.clone(), |ctx: Ctx<'js>, options: Object<'js>| {
         sheet_dialog(&ctx, &options)
      })?,
   )?;
   Ok(ui)
}

/// `phoenix.ui.choose` as a native Mana modal; the picked index, or -1 when it
/// was dismissed.
fn choose_dialog<'js>(
   ctx: &Ctx<'js>,
   options: &Object<'js>,
) -> rquickjs::Result<rquickjs::Promise<'js>> {
   let title = options.get::<_, Option<String>>("title")?;
   let message = options.get::<_, Option<String>>("message")?;
   let items = parse_choice_options(options)?;
   if message.is_none() && items.is_empty() {
      return Err(Exception::throw_type(
         ctx,
         "choose needs options or a message",
      ));
   }
   let body = message.map(|text| ModalSection::Text { title: None, text });
   let title = title.unwrap_or_else(|| "Sunbird".to_owned());
   in_background(ctx, move || answer(title, body, items))
}

/// `phoenix.ui.showSheet` as a native Mana modal with the code viewer.
fn sheet_dialog<'js>(
   ctx: &Ctx<'js>,
   options: &Object<'js>,
) -> rquickjs::Result<rquickjs::Promise<'js>> {
   let title = options.get::<_, Option<String>>("title")?;
   let text = options.get::<_, String>("text")?;
   let monospace = options
      .get::<_, Option<bool>>("monospace")?
      .unwrap_or_default();
   let highlight = options
      .get::<_, Option<bool>>("highlight")?
      .unwrap_or_default();
   let items = parse_choice_options(options)?;
   let body = ModalSection::Code {
      title: None,
      text,
      monospace,
      highlight,
   };
   let title = title.unwrap_or_else(|| "Sunbird".to_owned());
   in_background(ctx, move || answer(title, Some(body), items))
}

/// Opens a choose dialog and waits for the picked index, -1 when it is
/// dismissed or left open for ten minutes.
fn answer(
   title: String,
   body: Option<ModalSection>,
   items: Vec<modal::ChoiceOption>,
) -> Settlement {
   Settlement::Int(Ok(modal::ask(title, body, items, Duration::from_secs(600))))
}

fn parse_choice_options<'js>(options: &Object<'js>) -> rquickjs::Result<Vec<modal::ChoiceOption>> {
   Ok(options
      .get::<_, Option<Vec<Value<'js>>>>("options")?
      .unwrap_or_default()
      .into_iter()
      .filter_map(|option| {
         if let Some(label) = option.as_string() {
            return Some(modal::ChoiceOption {
               label:  label.to_string().ok()?,
               danger: false,
            });
         }
         let option = option.as_object()?;
         Some(modal::ChoiceOption {
            label:  option.get::<_, Option<String>>("label").ok().flatten()?,
            danger: option
               .get::<_, Option<bool>>("danger")
               .ok()
               .flatten()
               .unwrap_or_default(),
         })
      })
      .collect())
}

/// Runs registered `phoenix.messages.onSend` hooks against the JSON payload
/// of an outgoing `POST /channels/<channel_id>/messages` request on desktop.
pub fn intercept_send_body(engine: &Engine, channel_id: &str, body: &[u8]) -> RequestAction {
   engine.with(|ctx| {
      let Some(registry) = ctx.userdata::<Registry<'_>>() else {
         return RequestAction::Pass;
      };
      let interceptors = registry
         .interceptors
         .borrow()
         .iter()
         .map(|(_, plugin, function)| (plugin.clone(), function.clone()))
         .collect::<Vec<_>>();
      drop(registry);
      if interceptors.is_empty() {
         return RequestAction::Pass;
      }
      let Ok(mut payload) = serde_json::from_slice::<serde_json::Value>(body) else {
         return RequestAction::Pass;
      };
      let Some(original) = payload
         .get("content")
         .and_then(serde_json::Value::as_str)
         .map(str::to_owned)
      else {
         return RequestAction::Pass;
      };
      let Ok(argument) = Object::new(ctx.clone()) else {
         return RequestAction::Pass;
      };
      let _ = argument.set("channelId", channel_id);

      let mut value = original.clone();
      for (plugin, function) in interceptors {
         match function
            .call::<_, Value<'_>>((value.clone(), argument.clone()))
            .catch(&ctx)
         {
            Ok(result) if result.as_bool() == Some(false) => {
               return RequestAction::Drop;
            },
            Ok(result) => {
               if let Some(text) = result.as_string().and_then(|text| text.to_string().ok()) {
                  value = text;
               }
            },
            Err(err) => log::error!("[{plugin}] onSend: {err}"),
         }
      }
      if value == original {
         return RequestAction::Pass;
      }
      if let Some(obj) = payload.as_object_mut() {
         obj.insert("content".to_owned(), serde_json::Value::String(value));
      }
      serde_json::to_vec(&payload).map_or(RequestAction::Pass, RequestAction::Replace)
   })
}

/// Counts the handler toward the emoji unlock. Desktop has no picker presses
/// to give it: the unlocked picker inserts the emoji itself.
fn on_emoji_pick<'js>(ctx: &Ctx<'js>, plugin: &str) -> rquickjs::Result<Function<'js>> {
   static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

   let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
   let registry = ctx
      .userdata::<Desktop<'js>>()
      .ok_or_else(|| Exception::throw_internal(ctx, "no registry"))?;
   registry
      .emoji_pickers
      .borrow_mut()
      .push((id, plugin.to_owned()));
   emoji::unlock(true);
   drop(registry);
   Function::new(ctx.clone(), move |ctx: Ctx<'js>| {
      if let Some(registry) = ctx.userdata::<Desktop<'js>>() {
         let mut pickers = registry.emoji_pickers.borrow_mut();
         pickers.retain(|(existing, _)| *existing != id);
         emoji::unlock(!pickers.is_empty());
      }
   })
}

pub fn on_send<'js>(
   ctx: &Ctx<'js>,
   plugin: &str,
   function: Function<'js>,
) -> rquickjs::Result<Function<'js>> {
   static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

   let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
   let registry = ctx
      .userdata::<Registry<'js>>()
      .ok_or_else(|| Exception::throw_internal(ctx, "no registry"))?;
   registry
      .interceptors
      .borrow_mut()
      .push((id, plugin.to_owned(), function));
   drop(registry);
   Function::new(ctx.clone(), move |ctx: Ctx<'js>| {
      if let Some(registry) = ctx.userdata::<Registry<'js>>() {
         registry
            .interceptors
            .borrow_mut()
            .retain(|(existing, ..)| *existing != id);
      }
   })
}
