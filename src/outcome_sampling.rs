use std::{collections::HashMap, hash::Hash};

use rand::{prelude::SliceRandom, thread_rng};

use super::{ImperfectInfoGame, Solution};

struct ActionInfo {
  regret: f64,
  strategy_avg: f64,
}

impl ActionInfo {
  fn new() -> Self {
    Self {
      regret: 0.0,
      strategy_avg: 0.0,
    }
  }
}

struct InfosetInformation<A: Hash> {
  action_info: HashMap<A, ActionInfo>,
  last_update: usize,
}

impl<A: Hash> InfosetInformation<A> {
  fn new() -> Self {
    Self {
      action_info: HashMap::new(),
      last_update: 0,
    }
  }
}

pub enum SampleStrategy {
  EpsilonGreedy(f64),
}

#[derive(Debug)]
struct TraversedGameState<A> {
  action: A,
  sample_prob: f64,
  policy_prob: f64,
  turn: usize,
}

pub struct OutcomeSampling<I: Hash, A: Hash> {
  infoset_info: HashMap<I, InfosetInformation<A>>,
  update_index: usize,
  sample_strategy: SampleStrategy,
}

impl<I: Hash + Eq + Clone, A: Hash + Eq + Clone> OutcomeSampling<I, A> {
  pub fn new() -> Self {
    Self {
      infoset_info: HashMap::new(),
      update_index: 0,
      sample_strategy: SampleStrategy::EpsilonGreedy(0.5),
    }
  }

  pub fn solve_iter(
    &mut self,
    game: &mut dyn ImperfectInfoGame<Infoset = I, Action = A>,
  ) {
    self.update_index += 1;
    let update_index = self.update_index;
    for p in 0..game.num_players() {
      game.reset();
      let mut game_states = Vec::new();
      // Sample action sequence
      while !game.is_game_over() {
        let infoset = game.current_infoset();
        let actions = game.actions();
        let num_actions = actions.len();
        let probs = self.action_probs(infoset, actions).1;

        let sample_probs: Vec<_> = match self.sample_strategy {
          SampleStrategy::EpsilonGreedy(epsilon) => probs
            .iter()
            .map(|(_, p)| {
              *p * (1.0 - epsilon) + 1.0 / num_actions as f64 * epsilon
            })
            .collect(),
        };
        let prob_indices: Vec<_> = (0..probs.len()).collect();
        let action_ind = *prob_indices
          .choose_weighted(&mut thread_rng(), |i| sample_probs[*i])
          .unwrap();
        let action = probs[action_ind].0.clone();
        let policy_prob = probs[action_ind].1;
        game_states.push(TraversedGameState {
          action: action.clone(),
          sample_prob: sample_probs[action_ind],
          policy_prob,
          turn: game.turn(),
        });
        game.make_move(action);
      }

      // update relevant infosets
      let player_util = game.utility()[p];
      let pi_neg_i: Vec<_> = game_states
        .iter()
        .scan(1.0, |state, s| {
          let result = *state;
          if s.turn != p {
            *state *= s.policy_prob;
          }
          Some(result)
        })
        .collect();
      let pi_i: Vec<_> = game_states
        .iter()
        .scan(1.0, |state, s| {
          let result = *state;
          if s.turn == p {
            *state *= s.policy_prob;
          }
          Some(result)
        })
        .collect();
      let overall_sample_prob: f64 =
        game_states.iter().map(|s| s.sample_prob).product();
      let mut reach_prob = 1.0;
      for (i, state) in game_states.iter().enumerate().rev() {
        game.undo_move();
        let prev_reach_prob = reach_prob;
        reach_prob *= state.policy_prob;
        if game.turn() != p {
          continue;
        }
        let (infoset_metadata, action_probs) =
          self.action_probs(game.current_infoset(), game.actions());

        for (act, prob) in action_probs {
          let w_cap = player_util * pi_neg_i[i] / overall_sample_prob;
          let new_regret = if act == state.action {
            w_cap * (prev_reach_prob - reach_prob)
          } else {
            -w_cap * reach_prob
          };
          let action_info = infoset_metadata
            .action_info
            .entry(act.clone())
            .or_insert_with(ActionInfo::new);
          action_info.regret += new_regret;
          action_info.strategy_avg +=
            (update_index - infoset_metadata.last_update) as f64
              * pi_i[i]
              * prob;
        }
        infoset_metadata.last_update = update_index;
      }
    }
  }

  fn action_probs(
    &mut self,
    infoset: I,
    actions: Vec<A>,
  ) -> (&mut InfosetInformation<A>, Vec<(A, f64)>) {
    let infoset_metadata = self
      .infoset_info
      .entry(infoset)
      .or_insert_with(InfosetInformation::new);
    let mut probs: Vec<_> = actions
      .into_iter()
      .map(|act| {
        let action_info = infoset_metadata
          .action_info
          .entry(act.clone())
          .or_insert_with(ActionInfo::new);
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
}

impl<I: Hash + Eq + Clone, A: Hash + Eq + Clone> Solution<I, A>
  for OutcomeSampling<I, A>
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
