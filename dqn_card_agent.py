import math
from os import stat
import random
from collections import Counter, namedtuple, deque
from itertools import product, combinations
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from bot import Bot, Actions
import time

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# episode_durations = []
#
#
# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(
#         tuple(map(lambda s: s is not None, batch.next_state)),
#         device=device,
#         dtype=torch.bool,
#     )
#     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     state_action_values = policy_net(state_batch).gather(1, action_batch)
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1).values
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     with torch.no_grad():
#         next_state_values[non_final_mask] = (
#             target_net(non_final_next_states).max(1).values
#         )
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     # In-place gradient clipping
#     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#     optimizer.step()

HAND_SIZE = 8
MAX_CARDS = 5
SUITS = ["Diamonds", "Clubs", "Hearts", "Spades"]
RANKS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "Jack",
    "Queen",
    "King",
    "Ace",
]
N_CARDS = len(SUITS) * len(RANKS)
N_OBSERVATIONS = N_CARDS

MAX_CARDS_PER_HAND = 5
PLAY_OPTIONS = [Actions.PLAY_HAND, Actions.DISCARD_HAND]
N_ACTIONS = N_CARDS * MAX_CARDS_PER_HAND + len(PLAY_OPTIONS)
print(N_ACTIONS)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 32
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TAU = 0.05
LR = 1e-4


class DQNPlayBot(Bot):
    def __init__(
        self,
        deck: str,
        stake: int = 1,
        seed: str | None = None,
        challenge: str | None = None,
        bot_port: int = 12346,
    ):
        super().__init__(deck, stake, seed, challenge, bot_port)

        self.steps_done = 0
        self.policy_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.target_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(capacity=1000)
        self.last_state = None
        self.last_action = None
        self.last_score = 0

    def skip_or_select_blind(self, G):
        return [Actions.SELECT_BLIND]

    # with the extra complexity this brings I'm starting to think it *might* be stupid
    def card_to_int(self, suit, rank):
        return SUITS.index(suit) * len(RANKS) + RANKS.index(rank)

    def int_to_card(self, n: int) -> tuple[str, str]:
        return SUITS[n // len(RANKS)], RANKS[n % len(RANKS)]

    def hand_to_ints(self) -> list[int]:
        return [
            self.card_to_int(card["suit"], card["value"]) for card in self.G["hand"]
        ]

    def random_action(self):
        hand = torch.tensor(self.hand_to_ints(), dtype=torch.float32)
        num_cards = random.randint(1, MAX_CARDS_PER_HAND)
        indices = torch.randperm(len(hand))[:num_cards]
        selection = torch.randint(0, N_CARDS, (1, MAX_CARDS_PER_HAND))
        selection[:, :num_cards] = hand[indices]
        return torch.cat(
            (
                selection,
                torch.randint(len(PLAY_OPTIONS), (1, 1)),
            ),
            dim=1,
        ).to(device)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                # the model outputs the cards it "wants" to play five times (can be any card in the deck)
                actions = self.policy_net(state)
                card_choices = (
                    actions[:, : N_CARDS * MAX_CARDS_PER_HAND]
                    .view(-1, MAX_CARDS_PER_HAND, N_CARDS)
                    .max(2)
                    .indices
                )
                option_choice = (
                    actions[:, N_CARDS * MAX_CARDS_PER_HAND :].max(1).indices
                )
                print(card_choices)
                print(option_choice)
                action = torch.cat((card_choices, option_choice.unsqueeze(1)), dim=1)

                return action
        else:
            return self.random_action()

    # attempt to generously convert what the model "wants" based on the actual hand
    # THIS MAY PRODUCE AN EMPTY HAND IF THE MODEL ATTEMPTS TO PLAY STUFF IT DOESN'T HAVE
    def action_to_command(self, tensor) -> list | None:
        hand = self.G["hand"]
        action = tensor[0]
        cards = action[:-1]
        option = action[-1]
        selection = []
        for card in cards:
            suit, rank = self.int_to_card(int(card.item()))
            try:
                hand_index = next(
                    (
                        i
                        for i, c in enumerate(hand)
                        if c["suit"] == suit and c["value"] == rank
                    )
                )
                # lua indexes start at 1 (guess how I found out)
                selection.append(hand_index + 1)
            except StopIteration:
                pass

        return [PLAY_OPTIONS[int(option.item())], selection]

    def validate_command(self, command) -> bool:
        """
        Perform basic sanity check on commands to make sure they select cards and don't attempt invalid discards
        """
        # empty command or attempting to discard without any available
        if not command[1] or (
            command[0] == Actions.DISCARD_HAND
            and self.G["current_round"]["discards_left"] == 0
        ):
            return False
        # duplicate cards in hand
        return not [item for item, count in Counter(command[1]).items() if count > 1]

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        print(f"perform optimize step...")
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        print(state_action_values)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros((BATCH_SIZE, N_ACTIONS), device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def learn(self, state, reward):
        reward = torch.tensor([reward], device=device)

        if reward < 0:
            state_tensor = None
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # Store the transition in memory
        if self.last_state is not None:
            self.memory.push(self.last_state, self.last_action, state_tensor, reward)

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def select_cards_from_hand(self, G):
        score = self.G["chips"]
        reward = score - self.last_score
        print(f"reward delta: {reward}")

        hand = self.hand_to_ints()
        enc_hand = F.one_hot(torch.tensor([hand]), num_classes=len(SUITS) * len(RANKS))
        # currently the state is just the state of the hand (multi-hot encoded)
        state = enc_hand.sum(dim=1).to(device, dtype=torch.float)
        # self.learn(state, reward)

        action = self.select_action(state)
        command = self.action_to_command(action)
        # if the model generates a stupid command, act randomly instead (sort of a penalty)
        while not self.validate_command(command):
            command = self.action_to_command(self.random_action())

        self.last_score = score
        self.last_state = state
        self.last_action = action
        print(f"making action: {command}")
        return command

    def select_shop_action(self, G):
        return [Actions.END_SHOP]

    def select_booster_action(self, G):
        return [Actions.SKIP_BOOSTER_PACK]

    def sell_jokers(self, G):
        return [Actions.SELL_JOKER, []]

    def rearrange_jokers(self, G):
        return [Actions.REARRANGE_JOKERS, []]

    def use_or_sell_consumables(self, G):
        return [Actions.USE_CONSUMABLE, []]

    def rearrange_consumables(self, G):
        return [Actions.REARRANGE_CONSUMABLES, []]

    def rearrange_hand(self, G):
        return [Actions.REARRANGE_HAND, []]


if __name__ == "__main__":
    attempts = 2

    bot = DQNPlayBot(
        deck="Blue Deck", stake=1, seed=None, challenge=None, bot_port=12348
    )
    bot.start_balatro_instance()
    time.sleep(10)

    for i in range(attempts):
        print(f"attempt: {i}")
        bot.run()
    bot.stop_balatro_instance()
