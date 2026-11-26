//! The desktop share of the `phoenix` object: `phoenix.ui` and message
//! accessories, and `phoenix.native` for the console.

pub mod native;
pub mod ui;

use std::{
   cell::RefCell,
   collections::HashMap,
};

use rquickjs::{
   Ctx,
   Function,
   JsLifetime,
};

use crate::{
   accessories,
   call_timer,
   context_menu,
   emoji,
   js::ui::{
      DesktopAccessory,
      DesktopAction,
   },
};

/// What plugins registered with the desktop UI, kept in the runtime's userdata
/// next to the core's registry.
#[derive(Default)]
pub struct Desktop<'js> {
   /// `phoenix.ui.addMessageAccessory` callbacks.
   pub accessories:   RefCell<Vec<DesktopAccessory<'js>>>,
   /// `phoenix.ui.addMessageAction` callbacks.
   pub actions:       RefCell<Vec<DesktopAction<'js>>>,
   /// Active `phoenix.ui.openPage` button callbacks, keyed by button ID.
   pub page_buttons:  RefCell<HashMap<u64, (u64, String, Function<'js>)>>,
   /// `phoenix.messages.onEmojiPick` handlers by id and plugin. Desktop has
   /// no picker presses to give them, only the unlock they turn on.
   pub emoji_pickers: RefCell<Vec<(u64, String)>>,
}

// SAFETY: every field is bound to 'js, and Changed swaps exactly that lifetime.
unsafe impl<'js> JsLifetime<'js> for Desktop<'js> {
   type Changed<'to> = Desktop<'to>;
}

/// Drops the accessories, context menu actions, open page callbacks, and emoji
/// pick handlers `name` registered and redraws Discord without them.
pub fn unload(ctx: &Ctx<'_>, name: &str) {
   call_timer::clear_format(name);
   let Some(registry) = ctx.userdata::<Desktop<'_>>() else {
      return;
   };
   let mut list = registry.accessories.borrow_mut();
   let before = list.len();
   list.retain(|entry| entry.plugin != name);
   if list.len() != before {
      accessories::set_active_count(list.len());
      accessories::clear_cache();
      accessories::invalidate_discord_windows();
   }
   drop(list);

   let mut actions = registry.actions.borrow_mut();
   let before_actions = actions.len();
   actions.retain(|entry| entry.plugin != name);
   if actions.len() != before_actions {
      let meta = actions.iter().map(DesktopAction::meta).collect();
      context_menu::sync_actions(meta);
   }
   drop(actions);

   registry
      .page_buttons
      .borrow_mut()
      .retain(|_, (_, plugin, _)| plugin != name);

   let mut pickers = registry.emoji_pickers.borrow_mut();
   pickers.retain(|(_, plugin)| plugin != name);
   emoji::unlock(!pickers.is_empty());
}
