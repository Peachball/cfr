use std::{collections::HashMap, hash::Hash};

use rand::{prelude::SliceRandom, thread_rng};

use super::{ImperfectInfoGame, Solution};

struct CfrActionInfo {
  regret: f64,
  strategy_avg: f64,
}

impl CfrActionInfo {
  fn new() -> Self {
    Self {
      regret: 0.0,
      strategy_avg: 0.0,
    }
  }
}

struct CfrInfosetInformation<A: Hash> {
  action_info: HashMap<A, CfrActionInfo>,
  last_update: usize,
}

impl<A: Hash> CfrInfosetInformation<A> {
  fn new() -> Self {
    Self {
      action_info: HashMap::new(),
      last_update: 0,
    }
  }
}

/// Implements external sampling CFR algorithm
///
/// Assumes each infoset has a well defined player turn -- e.g. different
/// players cannot have the same infoset
pub struct ExternalSampling<I: Hash, A: Hash> {
  infoset_info: HashMap<I, CfrInfosetInformation<A>>,
  update_index: usize,
}

impl<I: Hash + Eq + Clone, A: Hash + Eq + Clone> ExternalSampling<I, A> {
  pub fn new() -> Self {
    Self {
      infoset_info: HashMap::new(),
      update_index: 0,
    }
  }

  pub fn solve_iter(
    &mut self,
    game: &mut dyn ImperfectInfoGame<Infoset = I, Action = A>,
  ) {
    // run cfr update
    self.update_index += 1;
    for p in 0..game.num_players() {
      game.reset();
      self.solve_helper(p, game, 1.0);
    }
  }

  /// Arguments:
  ///
  /// Returns:
  ///   utility of current game node for the player given the random state
  ///   (sum u_i * pi_i^sigma(Z[i], z))
  ///   corresponding to equation 16 in the appendix
  fn solve_helper(
    &mut self,
    player: usize,
    game: &mut dyn ImperfectInfoGame<Infoset = I, Action = A>,
    player_reach_prob: f64,
  ) -> f64 {
    if game.is_game_over() {
      return game.utility()[player];
    }
    // Initialize variables
    let infoset = game.current_infoset();
    let action_probs = self.action_probs(infoset.clone(), game.actions()).1;
    let mut action_regrets = Vec::with_capacity(action_probs.len());
    let turn = game.turn();

    if player == turn {
      // compute regrets
      let mut partial_utility = 0.0;
      for (act, prob) in &action_probs {
        game.make_move(act.clone());
        let new_reach_prob = player_reach_prob * prob;
        let action_util = self.solve_helper(player, game, new_reach_prob);
        game.undo_move();
        partial_utility += action_util * prob;
        action_regrets.push(action_util);
      }

      // update strategy averages + regrets
      let update_index = self.update_index;
      let cur_infoset = self.get_infoset(infoset.clone());
      for (i, (act, prob)) in action_probs.iter().enumerate() {
        let act_inf = cur_infoset
          .action_info
          .entry(act.clone())
          .or_insert_with(CfrActionInfo::new);
        act_inf.regret += action_regrets[i] - partial_utility;
        act_inf.strategy_avg += prob
          * player_reach_prob
          * (update_index - cur_infoset.last_update) as f64;
      }
      cur_infoset.last_update = update_index;
      partial_utility
    } else {
      let random_act = action_probs
        .choose_weighted(&mut thread_rng(), |(_, p)| *p)
        .unwrap();
      game.make_move(random_act.0.clone());
      let partial_utility = self.solve_helper(player, game, player_reach_prob);
      game.undo_move();
      partial_utility
    }
  }

  fn get_infoset(&mut self, infoset: I) -> &mut CfrInfosetInformation<A> {
    self
      .infoset_info
      .entry(infoset.clone())
      .or_insert_with(CfrInfosetInformation::new)
  }

  fn action_probs(
    &mut self,
    infoset: I,
    actions: Vec<A>,
  ) -> (&mut CfrInfosetInformation<A>, Vec<(A, f64)>) {
    let infoset_metadata = self
      .infoset_info
      .entry(infoset)
      .or_insert(CfrInfosetInformation::new());
    let mut probs: Vec<_> = actions
      .into_iter()
      .map(|act| {
        let action_info = infoset_metadata
          .action_info
          .entry(act.clone())
          .or_insert(CfrActionInfo::new());
        (act, action_info.regret.max(0.0))
      })
      .collect();
    let normalizing_constant: f64 = probs.iter().map(|(_, p)| p).sum();
    if normalizing_constant == 0.0 {
      let num_actions = probs.len();
      for (_, p) in probs.iter_mut() {
        *p = 1.0 / num_actions as f64;
      }
    } else {
      for (_, p) in probs.iter_mut() {
        *p = *p / normalizing_constant;
      }
    }
    (infoset_metadata, probs)
  }

  pub fn num_infosets(&self) -> usize {
    self.infoset_info.len()
  }
}

impl<I: Hash + Eq + Clone, A: Hash + Eq + Clone> Solution<I, A>
  for ExternalSampling<I, A>
{
  fn get_optimal_action_dist(
    &self,
    game: &mut dyn ImperfectInfoGame<Infoset = I, Action = A>,
  ) -> Vec<(A, f64)> {
    let actions = game.actions();
    let num_actions = actions.len();
    if let Some(info) = self.infoset_info.get(&game.current_infoset()) {
      let normalizing_constant: f64 =
        info.action_info.values().map(|ai| ai.strategy_avg).sum();
      if normalizing_constant == 0.0 {
        return actions
          .into_iter()
          .map(|a| (a, 1.0 / num_actions as f64))
          .collect();
      }
      actions
        .into_iter()
        .map(|a| {
          let prob = if let Some(a_inf) = info.action_info.get(&a) {
            a_inf.strategy_avg
          } else {
            0.0
          };
          (a, prob / normalizing_constant)
        })
        .collect()
    } else {
      actions
        .into_iter()
        .map(|a| (a, 1.0 / num_actions as f64))
        .collect()
    }
  }
}
