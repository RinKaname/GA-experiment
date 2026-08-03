import os
from hammurabi_env import DemocraticHammurabi

class LLMDemocraticHammurabi:
    """
    A wrapper around our DemocraticHammurabi environment to facilitate
    Agentic AI / SLM testing via text prompts and tool calls.
    """
    def __init__(self, max_years=12):
        self.env = DemocraticHammurabi(max_years=max_years)
        self.state = self.env.reset()

    def check_decree(self, acres_to_buy, bushels_to_feed, acres_to_plant):
        """The Royal Accountant checks the math before execution."""
        # Unpack state
        year, pop, grain, land, land_price, f_app, w_app, e_app, yrs_to_elec = self.state

        # Calculate land transactions
        cost_of_land = acres_to_buy * land_price if acres_to_buy > 0 else 0
        revenue_from_land = abs(acres_to_buy) * land_price if acres_to_buy < 0 else 0

        # In our environment, planting costs 1 bushel per acre
        cost_of_planting = acres_to_plant * 1.0

        total_costs = cost_of_land + bushels_to_feed + cost_of_planting
        available_funds = grain + revenue_from_land
        projected_bushels = available_funds - total_costs

        # 1. Check if we have enough grain
        if projected_bushels < 0:
            error_msg = f"ACCOUNTANT ERROR: Sire, your decree costs {total_costs} bushels (Land: {cost_of_land}, Feed: {bushels_to_feed}, Seed: {cost_of_planting}). We only have {available_funds} available (including land sales). We are short by {abs(projected_bushels)} bushels. Please recalculate."
            return False, error_msg

        # 2. Check if we have enough land to plant
        projected_acres = land + acres_to_buy
        if projected_acres < acres_to_plant:
            error_msg = f"ACCOUNTANT ERROR: Sire, you ordered us to plant {acres_to_plant} acres, but we only own {projected_acres} acres of land! Please recalculate."
            return False, error_msg

        # 3. Check if we have enough people to plant (1 person plants 10 acres max)
        max_plantable = pop * 10
        if acres_to_plant > max_plantable:
            error_msg = f"ACCOUNTANT ERROR: Sire, our {pop} people can only plant a maximum of {max_plantable} acres, but you ordered {acres_to_plant}. Please recalculate."
            return False, error_msg

        return True, "The math is sound, Sire."

    def play_turn(self, acres_to_buy, bushels_to_feed, acres_to_plant):
        """
        Takes raw numbers from the SLM tool call, converts them to the fractions
        expected by the underlying environment, and steps the game.
        """
        year, pop, grain, land, land_price, f_app, w_app, e_app, yrs_to_elec = self.state

        # The environment expects fractions. We must reverse-engineer the SLM's raw numbers into fractions.

        # 1. Land Action (-1 to 1)
        if acres_to_buy < 0:
            # Sell fraction: -acres_sold / total_land
            action_land = acres_to_buy / max(1.0, float(land))
        elif acres_to_buy > 0:
            # Buy fraction: acres_bought / max_affordable
            max_affordable = (grain) // land_price
            action_land = acres_to_buy / max(1.0, float(max_affordable))
        else:
            action_land = 0.0

        # We need to simulate the grain update for the fraction math of feed/plant
        projected_grain = grain
        if acres_to_buy < 0:
            projected_grain += abs(acres_to_buy) * land_price
        elif acres_to_buy > 0:
            projected_grain -= acres_to_buy * land_price

        # 2. Feed Action (fraction of projected total grain)
        action_feed = bushels_to_feed / max(1.0, float(projected_grain))
        projected_grain -= bushels_to_feed

        # 3. Plant Action (fraction of remaining grain)
        action_plant = acres_to_plant / max(1.0, float(projected_grain))

        actions = [action_land, action_feed, action_plant]

        # Step environment
        self.state, reward, done, info = self.env.step(actions)
        return done, info['reason']

    def get_slm_payload(self):
        """The data block passed to the SLM to generate the next turn's story."""
        year, pop, grain, land, land_price, f_app, w_app, e_app, yrs_to_elec = self.state

        # Calculate optimal feeding for the SLM to help it out slightly
        optimal_food = pop * 20

        return f"""
        REPORT:
        Year: {year}
        Years Until Next Election: {yrs_to_elec}

        RESOURCES:
        Population: {pop}
        Acres of Land: {land}
        Bushels in Storage: {grain}
        Current Land Price: {land_price} bushels/acre
        (NOTE: Your people require {optimal_food} bushels to avoid starvation this year).

        POLITICAL POLLS (Must stay above 45% average to win election):
        Farmers Approval: {f_app:.1f}%
        Workers Approval: {w_app:.1f}%
        Elites Approval: {e_app:.1f}%
        """

def issue_decree(acres_to_buy: int, bushels_to_feed: int, acres_to_plant: int):
    """
    Issues the royal decrees for the year, deciding the fate of Babylon.

    Args:
        acres_to_buy: The number of acres to buy (use a negative number to sell land).
        bushels_to_feed: The number of bushels to distribute to the populace for food.
        acres_to_plant: The number of acres to plant with seed for the next harvest.
    """
    pass # This function is only for schema generation

def get_system_prompt():
    metamemory_addition = ""
    if os.path.exists("metamemory.txt"):
        with open("metamemory.txt", "r") as f:
            past_lessons = f.read()
        metamemory_addition = f"\n\nPAST LIVES METAMEMORY (Learn from past mistakes):\n{past_lessons}\n"

    return f"""You are the Grand Vizier of Democratic Babylon.
I will provide you with a 'REPORT' containing raw data about the kingdom.
1. Think silently about the implications of the data, your budget, and the political polls.
2. Calculate your budget based on the Laws of Babylon.
3. Once you have reasoned through your strategy, you MUST call the `issue_decree` tool to finalize your decisions.
4. You must output your tool call using EXACTLY this syntax, replacing the values with your calculated numbers:
<|tool_call>call:issue_decree{{acres_to_buy: [number], bushels_to_feed: [number], acres_to_plant: [number]}}<tool_call|>

THE LAWS OF BABYLON (GAME MECHANICS):
Before making your decrees, you MUST calculate your budget in your thought block using these exact rules:
1. FEEDING: 1 person requires exactly 20 bushels to survive the year. If you feed them less, people will starve. Starvation drastically lowers Worker approval and causes instant impeachment if too high!
2. PLANTING: It costs exactly 1 bushel of seed to plant 1 acre of land. You cannot plant more acres than you own. 1 person can plant a maximum of 10 acres.
3. REAL ESTATE: Buying 1 acre costs the current 'Land Price' in bushels. Selling 1 acre (using a negative number for acres_to_buy) ADDS the 'Land Price' in bushels to your total available budget.
4. THE GOLDEN RULE: (Bushels spent on Buying Land) + (Bushels spent on Feeding) + (Bushels spent on Planting) MUST NOT exceed your total available Bushels (which includes any Bushels gained from Selling Land).

THE RULES OF POLITICS (HOW TO WIN):
1. ELECTIONS: An election occurs every 4 years. If your average approval drops below 45% (a minority coalition threshold), you will be impeached and lose the game.
2. FARMERS: Farmers love it when you buy land and plant seeds. They HATE it when you sell land.
3. WORKERS: Workers love when you have extra food, and they absolutely HATE starvation.
4. ELITES: Elites only care about total kingdom wealth. They want you to accumulate massive amounts of grain and land value.

You must balance these factions while surviving!{metamemory_addition}"""

if __name__ == "__main__":
    print(get_system_prompt())
    game = LLMDemocraticHammurabi()
    print(game.get_slm_payload())
