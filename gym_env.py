import math
import os
from queue import Queue
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

        self.steps_done = 0
        self.last_state = None
        self.last_action = None
        self.last_score = 0
        self.state_queue = Queue()
        self.command_queue = Queue()

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
        ranks = [card["value"] for card in hand]
        suits = [card["suit"] for card in hand]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # three-of-a-kind
        if 3 in rank_counts.values() and sorted(rank_counts.values()) != [2, 3]:
            return 10

        # full house: one three-of-a-kind and one pair
        if sorted(rank_counts.values()) == [2, 3]:
            return 20

        # Check for two pair (only if not full house)
        if list(rank_counts.values()).count(2) == 2:
            return 5

        # flush
        if any(count >= 5 for count in suit_counts.values()):
            return 15

        # Straight
        # could use better implementation for ace-low and ace-high strategies
        rank_order = {r: i for i, r in enumerate(RANKS, start=1)}
        sorted_ranks = sorted(rank_order[rank] for rank in set(ranks))
        # Look for consecutive sequences
        consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
                consecutive += 1
                if consecutive >= 5:
                    return 15
                    break
            else:
                consecutive = 1

        # Check for a single Pair
        if 2 in rank_counts.values():
            return -2

        # If none of the above, it's a High Card situation
        return -5

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

        # flush
        is_flush = any(count >= 5 for count in suit_counts.values())

        # straight (no ace-high/ace-low)
        is_straight = False
        rank_order = {r: i for i, r in enumerate(RANKS, start=1)}
        sorted_ranks = sorted(rank_order[rank] for rank in set(ranks))
        consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
                consecutive += 1
                if consecutive >= 5:
                    is_straight = True
                    break
            else:
                consecutive = 1

        # Determine hand type based on a hierarchy:
        # Full house, flush, straight, three-of-a-kind, two pair, pair, high card
        if sorted(rank_counts.values()) == [2, 3]:
            return "full_house"
        if is_flush:
            return "flush"
        if is_straight:
            return "straight"
        if 3 in rank_counts.values():
            return "three_of_a_kind"
        if list(rank_counts.values()).count(2) >= 2:
            return "two_pair"
        if 2 in rank_counts.values():
            return "pair"
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

    def build_choices_from_action(self, actions):
        card_choices = (
            actions[:, : N_CARDS * MAX_CARDS_PER_HAND]
            .view(-1, MAX_CARDS_PER_HAND, N_CARDS)
            .max(2)
            .indices
        )
        option_choice = actions[:, N_CARDS * MAX_CARDS_PER_HAND :].max(1).indices
        return torch.cat((card_choices, option_choice.unsqueeze(1)), dim=1)

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

        resource_bonus = scaling_factor * (
            (
                (start_discards - self.G["current_round"]["discards_left"])
                / start_discards
            )
            + ((start_hands - self.G["current_round"]["hands_left"]) / start_hands)
        )

        chip_reward = score - self.last_score

        # evaluate current hand, apply bonus to better hands (duh)
        hand_bonus = self.evaluate_hand(self.G["hand"])

        reward = max(chip_reward + resource_bonus + hand_bonus, 0)

        hand = self.hand_to_ints()
        enc_hand = F.one_hot(torch.tensor([hand]), num_classes=len(SUITS) * len(RANKS))
        # currently the state is just the state of the hand (multi-hot encoded)
        state = enc_hand.sum(dim=1).to(device, dtype=torch.float)

        advance_steps = True
        command = None
        while True:
            action = self.select_action(state, advance_steps)
            advance_steps = False
            command = self.action_to_command(action)

            self.last_score = score
            self.last_state = state
            self.last_action = action

            if self.validate_command(command):
                break
            else:
                print("Invalid action, applying penalty")
                reward = -15
                is_final = False

        # Log final reward
        self.writer.add_scalar("Reward/Final", reward, self.steps_done)
        print(f"Commiting action: {command}")
        return command

    def select_shop_action(self, G):
        global attempted_purchases
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
                    and card["label"] not in attempted_purchases
                ):
                    # logging.info(f"Attempting to buy specific card: {card}")
                    attempted_purchases.add(card["label"])  # Track attempted purchases
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
