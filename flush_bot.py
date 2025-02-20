from bot import Bot, Actions
from gamestates import cache_state
import time

import logging

# Set up logging for shop interactions and general debugging
logging.basicConfig(
    filename="flush_bot_log.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Tracks previously attempted purchases to avoid repeated buys
attempted_purchases = set()


# Plays flushes if possible
# otherwise keeps the most common suit
# Discarding the rest, or playing the rest if there are no discards left
class FlushBot(Bot):

    def skip_or_select_blind(self, G):
        cache_state("skip_or_select_blind", G)
        return [Actions.SELECT_BLIND]

    def select_cards_from_hand(self, G):
        logging.debug("Entering select_cards_from_hand")
        try:
            action = self._select_cards_from_hand(G)
            logging.debug(f"Action taken: {action}")
            return action
        except Exception as e:
            logging.error(f"Error in select_cards_from_hand: {e}")
            return [Actions.PLAY_HAND, [1]]  # Fallback action

    def _select_cards_from_hand(self, G):
        try:
            logging.debug(f"Game state before selecting cards: {G}")

            if "hand" not in G or not G["hand"]:
                logging.error("Hand is empty or missing from game state!")
                return [Actions.PLAY_HAND, [1]]  # Fallback

            suit_count = {suit: 0 for suit in ["Hearts", "Diamonds", "Clubs", "Spades"]}
            for card in G["hand"]:
                suit_count[card["suit"]] += 1

            most_common_suit = max(suit_count, key=suit_count.get)
            most_common_suit_count = suit_count[most_common_suit]

            if most_common_suit_count >= 5:
                flush_cards = [card for card in G["hand"] if card["suit"] == most_common_suit]
                flush_cards.sort(key=lambda x: x["value"], reverse=True)
                indices = [G["hand"].index(card) + 1 for card in flush_cards[:5]]
                action = [Actions.PLAY_HAND, indices]

                if any(i > len(G["hand"]) for i in indices):
                    logging.error(f"Invalid indices in PLAY_HAND action: {indices}")
                    return [Actions.PLAY_HAND, [1]]  # Fallback
                
                logging.debug(f"Playing flush: {action}")
                return action

            # Discarding non-suited cards
            discards = [card for card in G["hand"] if card["suit"] != most_common_suit]
            discards.sort(key=lambda x: x["value"], reverse=True)
            discards = discards[:5]
            indices = [G["hand"].index(card) + 1 for card in discards]

            if len(discards) > 0:
                action_type = Actions.DISCARD_HAND if G["current_round"]["discards_left"] > 0 else Actions.PLAY_HAND
                action = [action_type, indices]

                if any(i > len(G["hand"]) for i in indices):
                    logging.error(f"Invalid indices in DISCARD_HAND action: {indices}")
                    return [Actions.PLAY_HAND, [1]]  # Fallback
                
                logging.debug(f"Discarding non-suited cards: {action}")
                return action

            logging.warning("No valid flush or discard, playing first card by default.")
            return [Actions.PLAY_HAND, [1]]

        except Exception as e:
            logging.error(f"Exception in _select_cards_from_hand: {e}")
            return [Actions.PLAY_HAND, [1]]  # Fallback action

    def select_shop_action(self, G):
        global attempted_purchases
        logging.info(f"Shop state received: {G}")

        specific_joker_cards = {
        "Joker", "Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker",
 "Droll Joker",
        "Crafty Joker", "Joker Stencil", "Banner", "Mystic Summit", "Loyalty Card", 
        "Misprint", "Raised Fist", "Fibonacci", "Scary Face", "Abstract Joker", 
        "Pareidolia", "Gros Michel", "Even Steven", "Odd Todd", "Scholar", "Supernova",  "Burglar", "Blackboard", "Ice Cream", "Hiker", "Green Joker", 
        "Cavendish", "Card Sharp", "Red Card", "Hologram", "Baron", "Midas Mask", "Photograph", 
        "Erosion", "Baseball Card", "Bull", "Popcorn", "Ancient Joker", "Ramen", "Walkie Talkie", "Seltzer", "Castle", "Smiley Face", 
        "Acrobat", "Sock and Buskin", "Swashbuckler", "Bloodstone", "Arrowhead", "Onyx Agate", "Showman", 
        "Flower Pot", "Blueprint", "Wee Joker", "Merry Andy", "The Idol", "Seeing Double", "Hit the Road", "The Tribe", "Stuntman", "Brainstorm", "Shoot the Moon", 
        "Bootstraps", "Triboulet", "Yorik", "Chicot"
        }

        if "shop" in G and "dollars" in G:
            dollars = G["dollars"]
            cards = G["shop"]["cards"]
            logging.info(f"Current dollars: {dollars}, Available cards: {cards}")

            for i, card in enumerate(cards):
                if card["label"] in specific_joker_cards and card["label"] not in attempted_purchases:
                    logging.info(f"Attempting to buy specific card: {card}")
                    attempted_purchases.add(card["label"])  # Track attempted purchases
                    return [Actions.BUY_CARD, [i + 1]]

        logging.info("No specific joker cards found or already attempted. Ending shop interaction.")
        return [Actions.END_SHOP]

 

    def select_booster_action(self, G):
        return [Actions.SKIP_BOOSTER_PACK]

    def sell_jokers(self, G):
        if len(G["jokers"]) > 3:
            return [Actions.SELL_JOKER, [2]]

        return [Actions.SELL_JOKER, []]

    def rearrange_jokers(self, G):
        return [Actions.REARRANGE_JOKERS, []]

    def use_or_sell_consumables(self, G):
        return [Actions.USE_CONSUMABLE, []]

    def rearrange_consumables(self, G):
        return [Actions.REARRANGE_CONSUMABLES, []]

    def rearrange_hand(self, G):
        return [Actions.REARRANGE_HAND, []]


def benchmark_multi_instance():
    global t
    t = 0
    global first_time
    first_time = None

    # Benchmark the game states per second for different bot counts
    bot_counts = [1] # range(1, 21, 3)
    for bot_count in bot_counts:
        target_t = 50 * bot_count
        t = 0
        first_time = None

        bots = []
        for i in range(bot_count):
            mybot = FlushBot(
                deck="Blue Deck",
                stake=1,
                seed=None,
                challenge=None,
                bot_port=12348 + i,
            )

            bots.append(mybot)

        try:
            for bot in bots:
                bot.start_balatro_instance()
            time.sleep(20)

            start_time = time.time()
            while t < target_t:
                for bot in bots:
                    bot.run_step()
            end_time = time.time()

            t_per_sec = target_t / (end_time - start_time)
            print(f"Bot count: {bot_count}, t/sec: {t_per_sec}")
        finally:
            # Stop the bots
            for bot in bots:
                bot.stop_balatro_instance()


if __name__ == "__main__":
    benchmark_multi_instance()
