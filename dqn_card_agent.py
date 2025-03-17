import math
import os
import random
from collections import Counter, namedtuple, deque
import random
import time
from pathlib import Path
from datetime import datetime
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from bot import Bot, Actions
import datetime
import time

# Tensorboard logging folder
run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
LOG_DIR = f"runs/dqn_balatro_{run_id}"

# Checkpoint folder
# TODO: dynamic checkpoint saving instead of a static path
LOAD_CHECKPOINT = False
# Important: Last checkpoint will be written over
SAVE_CHECKPOINT = True
CHECKPOINT_PATH = "checkpoints/checkpoint.pth"

# amount of steps between saving replays
CHECKPOINT_STEPS = 2500

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
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, 512)
        self.layer5 = nn.Linear(512, 256)
        self.layer6 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)


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

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
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

        self.hand_counts = {
            "high_card": 0,
            "pair": 0,
            "two_pair": 0,
            "three_of_a_kind": 0,
            "straight": 0,
            "flush": 0,
            "full_house": 0,
        }

        # tensorboard logging
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.writer = SummaryWriter(log_dir=f"runs/dqn_balatro_{run_id}")

        self.steps_done = 0
        self.policy_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.target_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(capacity=2500)
        self.last_state = None
        self.last_action = None
        self.last_command = None
        self.last_score = 0

        # Tracks previously attempted purchases to avoid repeated buys
        self.attempted_purchases = set()

        if LOAD_CHECKPOINT and CHECKPOINT_PATH is not None:
            self.load_checkpoint(CHECKPOINT_PATH)

    def save_checkpoint(self, step):
        checkpoint = {
            "steps_done": self.steps_done,
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_memory": list(self.memory.memory),
        }
        Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Checkpoint saved at step {step} to {CHECKPOINT_PATH}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        self.steps_done = checkpoint.get("steps_done", 0)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load replay memory: convert the saved list back to a deque with the same capacity.
        replay_memory_list = checkpoint.get("replay_memory", [])
        self.memory.memory = deque(replay_memory_list, maxlen=self.memory.memory.maxlen)

        print(f"Checkpoint loaded from {checkpoint_path}")

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

    @staticmethod
    def check_straights(ranks, rank_order):
        sorted_ranks = sorted(rank_order[rank] for rank in set(ranks))
        # Look for consecutive sequences
        consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
                consecutive += 1
                if consecutive >= 5:
                    return True
            else:
                consecutive = 1
        return False

    def evaluate_hand(self, hand):
        """
        Evaluates the hand for poker combinations.
        Expects hand as a list of card dictionaries, e.g. {"suit": "Hearts", "value": "Ace"}.
        Returns a bonus reward based on the hand quality.
        Hand reward hierarchy:
        - Full House: +20
        - Flush: +15
        - Straight: +15
        - Three-of-a-Kind: +10
        - Two Pair: +5
        - Pair: -2 (penalty)
        - High Card: -5 (penalty)
        """
        VALUE_MAP = {
            "full_house": 20,
            "three_of_a_kind": 10,
            "two_pair": 5,
            "flush": 15,
            "straight": 15,
            "pair": -2,
            "high_card": -5,
        }

        hand_type = self.classify_hand(hand)
        return VALUE_MAP[hand_type]

    def classify_hand(self, hand):
        """
        To keep track of how the agent is selecting hands.
        Classifies the hand into one of these categories:
        "full_house", "flush", "straight", "three_of_a_kind", "two_pair", "pair", or "high_card".
        Expects hand as a list of card dictionaries, e.g. {"suit": "Hearts", "value": "Ace"}.
        """
        ranks = [card["value"] for card in hand]
        suits = [card["suit"] for card in hand]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Determine hand type based on a hierarchy:
        # Full house, flush, straight, three-of-a-kind, two pair, pair, high card

        # straights
        # ace-high
        rank_order = {r: i for i, r in enumerate(RANKS, start=2)}
        if DQNPlayBot.check_straights(ranks, rank_order):
            return "straight"
        # ace-low
        rank_order["Ace"] = 1
        if DQNPlayBot.check_straights(ranks, rank_order):
            return "straight"

        # full house: one three-of-a-kind and one pair
        # flush
        if any(count >= 5 for count in suit_counts.values()):
            return "flush"
        elif all(count in rank_counts.values() for count in [2, 3]):
            return "full_house"
        # three-of-a-kind
        elif 3 in rank_counts.values():
            return "three_of_a_kind"
        # Check for two pair (only if not full house)
        elif list(rank_counts.values()).count(2) >= 2:
            return "two_pair"
        elif 2 in rank_counts.values():
            return "pair"
        else:
            return "high_card"

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

    def build_choices_from_action(self, actions, hand_encoded):
        card_choices = torch.multinomial(
            (
                actions[:, : N_CARDS * MAX_CARDS_PER_HAND]
                .view(-1, MAX_CARDS_PER_HAND, N_CARDS)
                .softmax(dim=2)
                * hand_encoded.unsqueeze(1)
            ).view(-1, N_CARDS),
            num_samples=1,
        )
        card_choices = card_choices.view(-1, MAX_CARDS_PER_HAND)
        option_choice = actions[:, N_CARDS * MAX_CARDS_PER_HAND :].max(1).indices
        return torch.cat((card_choices, option_choice.unsqueeze(1)), dim=1)

    def select_action(self, state, advance_steps=True):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.writer.add_scalar("Epsilon/Threshold", eps_threshold, self.steps_done)
        print(f"Current threshold: {eps_threshold}")
        if advance_steps:
            self.steps_done += 1
        if sample > eps_threshold:
            print("performing neural action...")
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                # the model outputs the cards it "wants" to play five times (can be any card in the deck)
                return self.build_choices_from_action(self.policy_net(state), state)
        else:
            return self.random_action()

    # attempt to generously convert what the model "wants" based on the actual hand
    # THIS MAY PRODUCE AN EMPTY HAND IF THE MODEL ATTEMPTS TO PLAY STUFF IT DOESN'T HAVE
    def action_to_command(self, tensor) -> list | None:
        hand = self.G["hand"]
        action = tensor[0]
        cards = action[:-1]
        option = action[-1]
        selection = set()
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
                selection.add(hand_index + 1)
            except StopIteration:
                pass

        return [PLAY_OPTIONS[int(option.item())], list(selection)]

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

    def gather_action_weights(self, action_tensor, action_batch):
        cards_action_values = (
            action_tensor[:, : N_CARDS * MAX_CARDS_PER_HAND]
            .view(-1, MAX_CARDS_PER_HAND, N_CARDS)
            .gather(
                2, action_batch[:, :MAX_CARDS_PER_HAND].view(-1, MAX_CARDS_PER_HAND, 1)
            )
            .view(-1, MAX_CARDS_PER_HAND)
        )
        option_action_values = action_tensor[:, N_CARDS * MAX_CARDS_PER_HAND :].gather(
            1, action_batch[:, MAX_CARDS_PER_HAND:]
        )
        return torch.cat((cards_action_values, option_action_values), dim=1)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        # print(f"perform optimize step...")
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
        state_action_values = self.gather_action_weights(
            self.policy_net(state_batch), action_batch
        )

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(
            (BATCH_SIZE, MAX_CARDS_PER_HAND + 1), device=device
        )
        with torch.no_grad():
            next_actions = self.target_net(non_final_next_states)
            next_choices = self.build_choices_from_action(
                next_actions, non_final_next_states
            )
            next_state_values[non_final_mask] = self.gather_action_weights(
                next_actions, next_choices
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Log loss value
        self.writer.add_scalar("Loss", loss.item(), self.steps_done)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def learn(self, state, reward, is_final):
        reward = torch.tensor([[reward]], device=device)

        if is_final:
            state_tensor = None
        else:
            state_tensor = state
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

        # Save checkpoint every N steps
        if self.steps_done % CHECKPOINT_STEPS == 0 and SAVE_CHECKPOINT:
            self.save_checkpoint(self.steps_done)

    def select_cards_from_hand(self, G):
        """
        Option 1:
        Tuning reward based on resource usage:
        If the agent has used many discards and hands, the term is high,
            suggesting that the agent took actions to improve the hand
        Conversely, if resources are hoarded when cards need to be discarded,
            the term remains low

        Heuristic given a fixed D discards and H hands
        Calculate resource usage as
        λ((D - discards_left)/D + (H - hands_left)/H)

        combining this with given chip rewards
        reward = (current_chips - last_chips) + resource_usage

        Option 2:
        Penalize leaving too many unused resouces when the hand quality is poor
        penalty = λ((discards_left)/D + (hands_left)/H)
        """
        # reward = max(score - self.last_score, 0)
        # reward = score - self.last_score

        # don't really want to store this in state
        # should store or grab from API
        start_discards = 3
        start_hands = 5
        scaling_factor = 5  # λ
        score = self.G["chips"]
        self.writer.add_scalar("Chip reward", score, self.steps_done)

        resource_bonus = scaling_factor * (
            (
                (start_discards - self.G["current_round"]["discards_left"])
                / start_discards
            )
            + ((start_hands - self.G["current_round"]["hands_left"]) / start_hands)
        )

        chip_reward = score - self.last_score
        is_final = chip_reward < 0

        # evaluate current hand, apply bonus to better hands (duh)
        if self.last_command is not None:
            last_play = [self.G["hand"][i - 1] for i in self.last_command[1]]
            hand_bonus = self.evaluate_hand(last_play)
        else:
            hand_bonus = 0

        reward = max(chip_reward, 0) + hand_bonus + resource_bonus
        self.writer.add_scalar("Reward/Delta", reward, self.steps_done)
        # print(f"reward delta: {reward}")

        # for logging only, classify hand
        hand_type = self.classify_hand(self.G["hand"])

        # increment the counter for this hand type
        if hand_type in self.hand_counts:
            self.hand_counts[hand_type] += 1

        self.writer.add_scalars("HandCounts", self.hand_counts, self.steps_done)

        hand = self.hand_to_ints()
        enc_hand = F.one_hot(torch.tensor([hand]), num_classes=len(SUITS) * len(RANKS))
        # currently the state is just the state of the hand (multi-hot encoded)
        state = enc_hand.sum(dim=1).to(device, dtype=torch.float)

        if self.steps_done % CHECKPOINT_STEPS == 0:
            self.save_checkpoint(self.steps_done)

        advance_steps = True
        command = None
        retries = 0
        while True:
            self.learn(state, reward, is_final)

            action = self.select_action(state, advance_steps)
            advance_steps = False
            command = self.action_to_command(action)

            self.last_score = score
            self.last_state = state
            self.last_action = action
            self.last_command = command

            if self.validate_command(command):
                break
            else:
                retries += 1
                reward = -17
                is_final = False

        self.writer.add_scalar("Action Retries", retries, self.steps_done)
        # Log final reward
        self.writer.add_scalar("Reward/Final", reward, self.steps_done)
        print(f"Commiting action: {command}")
        return command

    def select_shop_action(self, G):
        # logging.info(f"Shop state received: {G}")

        specific_joker_cards = {
            "Joker",
            "Greedy Joker",
            "Lusty Joker",
            "Wrathful Joker",
            "Gluttonous Joker",
            "Droll Joker",
            "Clever Joker",
            "Devious Joker",
            "The Duo",
            "The Trio",
            "The Family",
            "The Order",
            "Crafty Joker",
            "Joker Stencil",
            "Banner",
            "Mystic Summit",
            "Loyalty Card",
            "Jolly Joker",
            "Sly Joker",
            "Wily Joker",
            "Half Joker",
            "Spare Trousers",
            "Misprint",
            "Raised Fist",
            "Fibonacci",
            "Scary Face",
            "Abstract Joker",
            "Zany Joker",
            "Mad Joker",
            "Crazy Joker",
            "Four Fingers",
            "Runner",
            "Pareidolia",
            "Gros Michel",
            "Even Steven",
            "Odd Todd",
            "Scholar",
            "Supernova",
            "Burglar",
            "Blackboard",
            "Ice Cream",
            "Hiker",
            "Green Joker",
            "Cavendish",
            "Card Sharp",
            "Red Card",
            "Hologram",
            "Baron",
            "Midas Mask",
            "Photograph",
            "Erosion",
            "Baseball Card",
            "Bull",
            "Popcorn",
            "Ancient Joker",
            "Ramen",
            "Walkie Talkie",
            "Seltzer",
            "Castle",
            "Smiley Face",
            "Acrobat",
            "Sock and Buskin",
            "Swashbuckler",
            "Bloodstone",
            "Arrowhead",
            "Onyx Agate",
            "Showman",
            "Flower Pot",
            "Blueprint",
            "Wee Joker",
            "Merry Andy",
            "The Idol",
            "Seeing Double",
            "Hit the Road",
            "The Tribe",
            "Stuntman",
            "Brainstorm",
            "Shoot the Moon",
            "Bootstraps",
            "Triboulet",
            "Yorik",
            "Chicot",
        }

        if "shop" in G and "dollars" in G:
            dollars = G["dollars"]
            cards = G["shop"]["cards"]
            # logging.info(f"Current dollars: {dollars}, Available cards: {cards}")

            for i, card in enumerate(cards):
                if (
                    card["label"] in specific_joker_cards
                    and card["label"] not in self.attempted_purchases
                ):
                    # logging.info(f"Attempting to buy specific card: {card}")
                    self.attempted_purchases.add(
                        card["label"]
                    )  # Track attempted purchases
                    return [Actions.BUY_CARD, [i + 1]]

        # logging.info("No specific joker cards found or already attempted. Ending shop interaction.")
        return [Actions.END_SHOP]

    def select_booster_action(self, G):
        return [Actions.SKIP_BOOSTER_PACK]

    def sell_jokers(self, G):
        if len(G["jokers"]) > 3:
            return [Actions.SELL_JOKER, [2]]
        else:
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
        deck="Blue Deck", stake=1, seed=None, challenge=None, bot_port=12346
    )

    if len(sys.argv) >= 2:
        bot.load_checkpoint(Path(sys.argv[1]))

    bot.start_balatro_instance()
    time.sleep(10)

    for i in range(attempts):
        print(f"attempt: {i}")
        bot.run()
    bot.stop_balatro_instance()
    bot.writer.close()
