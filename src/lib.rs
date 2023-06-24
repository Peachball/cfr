use std::{collections::HashMap, hash::Hash, iter};

pub mod external_sampling;
pub mod outcome_sampling;

/// All invalid calls panic on failure
pub trait ImperfectInfoGame {
  type Infoset: Hash + Eq + Clone;
  type Action: Hash + Eq + Clone;

  fn reset(&mut self);
  fn current_infoset(&self) -> Self::Infoset;
  fn actions(&self) -> Vec<Self::Action>;
  fn make_move(&mut self, action: Self::Action);
  fn undo_move(&mut self);
  fn is_game_over(&self) -> bool;
  fn utility(&self) -> Vec<f64>;
  fn turn(&self) -> usize;
  fn num_players(&self) -> usize;
}

pub trait Solution<I, A> {
  fn get_optimal_action_dist(
    &self,
    game: &mut dyn ImperfectInfoGame<Infoset = I, Action = A>,
  ) -> Vec<(A, f64)>;
}

pub trait EnumerableImperfectInfoGame: ImperfectInfoGame {
  type Rng;
  fn reset_with_state(&mut self, state: Self::Rng);
  fn get_rng_iter(&self) -> Box<dyn Iterator<Item = Self::Rng>>;
}

pub struct RPS {
  p1_action: Option<RPSAction>,
  p2_action: Option<RPSAction>,
  utility_matrix: [[i32; 3]; 3],
}

#[derive(Hash, Clone, Copy, PartialEq, Eq, Debug)]
pub enum RPSAction {
  Rock,
  Paper,
  Scissors,
}

impl From<RPSAction> for usize {
  fn from(action: RPSAction) -> Self {
    match action {
      RPSAction::Rock => 0,
      RPSAction::Paper => 1,
      RPSAction::Scissors => 2,
    }
  }
}

impl RPS {
  pub fn new() -> Self {
    Self {
      p1_action: None,
      p2_action: None,
      utility_matrix: [[0, 2, -2], [-1, 0, 1], [1, -1, 0]],
    }
  }
}

impl ImperfectInfoGame for RPS {
  type Infoset = usize;
  type Action = RPSAction;

  fn reset(&mut self) {
    self.p1_action = None;
    self.p2_action = None;
  }

  fn current_infoset(&self) -> Self::Infoset {
    self.turn()
  }

  fn actions(&self) -> Vec<Self::Action> {
    vec![RPSAction::Rock, RPSAction::Scissors, RPSAction::Paper]
  }

  fn make_move(&mut self, action: Self::Action) {
    if self.p1_action.is_none() {
      self.p1_action = Some(action);
    } else {
      self.p2_action = Some(action);
    }
  }

  fn undo_move(&mut self) {
    if self.p2_action.is_some() {
      self.p2_action = None;
    } else {
      self.p1_action = None;
    }
  }

  fn is_game_over(&self) -> bool {
    self.p1_action.is_some() && self.p2_action.is_some()
  }

  fn utility(&self) -> Vec<f64> {
    let p1_util = self.utility_matrix[usize::from(self.p1_action.unwrap())]
      [usize::from(self.p2_action.unwrap())] as f64;
    vec![p1_util, -p1_util]
  }

  fn turn(&self) -> usize {
    if self.p1_action.is_none() {
      0
    } else {
      1
    }
  }

  fn num_players(&self) -> usize {
    2
  }
}

impl EnumerableImperfectInfoGame for RPS {
  type Rng = ();

  fn reset_with_state(&mut self, _: Self::Rng) {
    self.reset();
  }

  fn get_rng_iter(&self) -> Box<dyn Iterator<Item = Self::Rng>> {
    Box::new(iter::once(()))
  }
}

struct CfrActionInfo {
  regret: f64,
  utility: f64,
  strategy_avg: f64,
}

impl CfrActionInfo {
  fn new() -> Self {
    Self {
      regret: 0.0,
      utility: 0.0,
      strategy_avg: 0.0,
    }
  }

  fn reset_partial_values(&mut self) {
    self.utility = 0.0;
  }
}

struct CfrInfosetInformation<A: Hash> {
  utility_numerator: f64,
  pi_i: f64,
  pi_neg_i: f64,
  action_info: HashMap<A, CfrActionInfo>,
  strategy_avg_total: f64,
}

impl<A: Hash> CfrInfosetInformation<A> {
  fn new() -> Self {
    Self {
      utility_numerator: 0.0,
      pi_i: 0.0,
      pi_neg_i: 0.0,
      action_info: HashMap::new(),
      strategy_avg_total: 0.0,
    }
  }

  fn reset_partial_values(&mut self) {
    self.utility_numerator = 0.0;
    self.pi_i = 0.0;
    self.pi_neg_i = 0.0;
    for (_, action_info) in self.action_info.iter_mut() {
      action_info.reset_partial_values();
    }
  }
}

/// Implements the CFR algorithm
///
/// Assumes each infoset has a well defined player turn -- e.g. different
/// players cannot have the same infoset
pub struct Cfr<I: Hash, A: Hash> {
  infoset_info: HashMap<I, CfrInfosetInformation<A>>,
}

impl<I: Hash + Eq + Clone, A: Hash + Eq + Clone> Cfr<I, A> {
  pub fn new() -> Self {
    Self {
      infoset_info: HashMap::new(),
    }
  }

  pub fn solve_iter<R>(
    &mut self,
    game: &mut dyn EnumerableImperfectInfoGame<
      Infoset = I,
      Action = A,
      Rng = R,
    >,
  ) {
    // run cfr update
    let mut reach_probs = HashMap::new();
    for rng_state in game.get_rng_iter() {
      game.reset_with_state(rng_state);
      self.solve_helper(game, &mut reach_probs);
    }

    // update regrets
    for info in self.infoset_info.values_mut() {
      let pi_neg_i = info.pi_neg_i;
      if info.pi_neg_i == 0.0 {
        continue;
      }
      let infoset_util = info.utility_numerator / info.pi_neg_i;
      for act_inf in info.action_info.values_mut() {
        let regret_diff = act_inf.utility / info.pi_neg_i - infoset_util;
        act_inf.regret += pi_neg_i * regret_diff;
      }
    }

    // Accumulate average strategy
    for info in self.infoset_info.values_mut() {
      let total_regret_sum: f64 = info
        .action_info
        .values()
        .map(|act_inf| act_inf.regret.max(0.0))
        .sum();

      let total_actions = info.action_info.len() as f64;
      for act_inf in info.action_info.values_mut() {
        let strat_prob = if total_regret_sum == 0.0 {
          1.0 / total_actions
        } else {
          act_inf.regret.max(0.0) / total_regret_sum
        };
        act_inf.strategy_avg += info.pi_i * strat_prob;
      }
      info.strategy_avg_total += info.pi_i;

      // reset infoset
      info.reset_partial_values();
    }
  }

  /// Arguments:
  ///   - reach_probs is a mapping of player index to pi_i probability under the
  ///   current strategy profile
  ///
  /// Returns:
  ///   utility of current game node given the random state
  fn solve_helper<R>(
    &mut self,
    game: &mut dyn EnumerableImperfectInfoGame<
      Infoset = I,
      Action = A,
      Rng = R,
    >,
    reach_probs: &mut HashMap<usize, f64>,
  ) -> Vec<f64> {
    if game.is_game_over() {
      return game.utility();
    }
    // Initialize variables
    let actions = game.actions();
    let turn = game.turn();
    let infoset = game.current_infoset();
    let mut total_utility = Vec::with_capacity(game.num_players());
    total_utility.resize(game.num_players(), 0.0);
    let info = self.get_infoset(infoset.clone());
    let orig_player_reach_prob = *reach_probs.get(&turn).unwrap_or(&1.0);

    // Accumulate some information
    let reach_prob_neg_i = reach_probs
      .iter()
      .filter_map(
        |(player, prob)| if player == &turn { None } else { Some(prob) },
      )
      .product::<f64>();
    info.pi_neg_i += reach_prob_neg_i;
    info.pi_i += orig_player_reach_prob;

    // Compute positive regrets
    let actions_to_regrets: Vec<_> = actions
      .into_iter()
      .map(|act| {
        let regret = info
          .action_info
          .get(&act)
          .map(|ai| ai.regret)
          .unwrap_or(0.0)
          .max(0.0);
        (act, regret)
      })
      .collect();

    // compute strategy profile
    let total_node_regret: f64 =
      actions_to_regrets.iter().map(|(_, r)| r).sum();
    let weights: Vec<f64> = actions_to_regrets
      .iter()
      .map(|(_, regret)| {
        if total_node_regret == 0.0 {
          1.0 / actions_to_regrets.len() as f64
        } else {
          regret / total_node_regret
        }
      })
      .collect();
    for ((act, _), prob) in actions_to_regrets.iter().zip(weights.iter()) {
      reach_probs.insert(turn, orig_player_reach_prob * prob);
      game.make_move(act.clone());
      let partial_utility = self.solve_helper(game, reach_probs);
      game.undo_move();

      for (i, util) in total_utility.iter_mut().enumerate() {
        *util += prob * partial_utility[i];
      }
      let info = self.get_infoset(infoset.clone());
      info.utility_numerator += reach_prob_neg_i * prob * partial_utility[turn];
      let action_info = info
        .action_info
        .entry(act.clone())
        .or_insert_with(CfrActionInfo::new);
      action_info.utility += partial_utility[turn] * reach_prob_neg_i;
    }
    reach_probs.insert(turn, orig_player_reach_prob);
    total_utility
  }

  fn get_infoset(&mut self, infoset: I) -> &mut CfrInfosetInformation<A> {
    self
      .infoset_info
      .entry(infoset.clone())
      .or_insert_with(CfrInfosetInformation::new)
  }
}

impl<I: Hash + Eq + Clone, A: Hash + Eq + Clone> Solution<I, A> for Cfr<I, A> {
  fn get_optimal_action_dist(
    &self,
    game: &mut dyn ImperfectInfoGame<Infoset = I, Action = A>,
  ) -> Vec<(A, f64)> {
    if let Some(info) = self.infoset_info.get(&game.current_infoset()) {
      info
        .action_info
        .iter()
        .map(|(act, act_inf)| {
          (act.clone(), act_inf.strategy_avg / info.strategy_avg_total)
        })
        .collect()
    } else {
      vec![]
    }
  }
}
