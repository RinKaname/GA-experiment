import random
import math

class DemocraticHammurabi:
    def __init__(self, max_years=12):
        self.max_years = max_years
        self.reset()

    def reset(self):
        self.year = 1
        self.population = 100
        self.grain = 2800
        self.land = 1000
        self.land_price = random.randint(17, 26)

        # Faction approvals (0 to 100)
        self.farmers_approval = 50.0
        self.workers_approval = 50.0
        self.elites_approval = 50.0

        self.is_done = False
        self.game_over_reason = ""
        self.starved_total = 0

        return self._get_state()

    def _get_state(self):
        years_until_election = 4 - (self.year % 4)
        if years_until_election == 4:
            years_until_election = 0

        return [
            self.year,
            self.population,
            self.grain,
            self.land,
            self.land_price,
            self.farmers_approval,
            self.workers_approval,
            self.elites_approval,
            years_until_election
        ]

    def step(self, actions):
        """
        Actions should be a list of 3 continuous values:
        - action_land: [-1, 1]. <0 means sell fraction of land, >0 means buy fraction of max affordable land.
        - action_feed: [0, 1]. fraction of total grain to use for feeding people.
        - action_plant: [0, 1]. fraction of remaining grain to use for planting.
        """
        if self.is_done:
            return self._get_state(), 0, self.is_done, {"reason": "Already done"}

        action_land, action_feed, action_plant = actions

        # Clip actions to expected ranges
        action_land = max(-1.0, min(1.0, action_land))
        action_feed = max(0.0, min(1.0, action_feed))
        action_plant = max(0.0, min(1.0, action_plant))

        # 1. Buy or Sell Land
        land_changed = 0
        if action_land < 0:
            # Sell land
            acres_to_sell = int(abs(action_land) * self.land)
            self.land -= acres_to_sell
            self.grain += acres_to_sell * self.land_price
            land_changed = -acres_to_sell
        elif action_land > 0:
            # Buy land
            max_acres_affordable = self.grain // self.land_price
            acres_to_buy = int(action_land * max_acres_affordable)
            self.land += acres_to_buy
            self.grain -= acres_to_buy * self.land_price
            land_changed = acres_to_buy

        # 2. Feed People
        grain_for_food = int(action_feed * self.grain)
        self.grain -= grain_for_food

        people_fed = grain_for_food // 20
        starved = max(0, self.population - people_fed)
        self.starved_total = starved

        if starved > 0:
            self.population -= starved

        # Immediate game over if starvation is extreme (>45%)
        if starved > 0.45 * (self.population + starved):
            self.is_done = True
            self.game_over_reason = "Impeached for extreme starvation"
            return self._get_state(), self._calculate_reward(), self.is_done, {"reason": self.game_over_reason}

        # 3. Plant Seeds
        grain_for_planting = int(action_plant * self.grain)
        max_plantable_by_people = self.population * 10

        # You need 1 grain per acre
        actual_planted = min(grain_for_planting, self.land, max_plantable_by_people)
        self.grain -= actual_planted

        # 4. Harvest & Rats
        yield_per_acre = random.randint(1, 5)
        harvest = actual_planted * yield_per_acre
        self.grain += harvest

        rats_ate = 0
        if random.random() < 0.4: # 40% chance of rats
            rats_ate = int(self.grain * random.uniform(0.1, 0.3))
            self.grain -= rats_ate

        # 5. Demographics (Births & Immigrants)
        immigrants = 0
        if starved == 0:
            immigrants = random.randint(1, 10) + int((20 * self.land + self.grain) / (100 * self.population + 1))
            self.population += immigrants

        # 6. Faction Approval Updates
        if land_changed > 0:
            self.farmers_approval += 5
        elif land_changed < 0:
            self.farmers_approval -= 10
        if actual_planted == self.land:
            self.farmers_approval += 5

        if starved > 0:
            self.workers_approval -= (starved / self.population) * 100
        else:
            self.workers_approval += 5

        total_wealth = self.grain + (self.land * self.land_price)
        expected_wealth = 2800 + (1000 * 20)
        if total_wealth > expected_wealth:
            self.elites_approval += 5
        else:
            self.elites_approval -= 5

        self.farmers_approval = self._clamp_and_decay_approval(self.farmers_approval)
        self.workers_approval = self._clamp_and_decay_approval(self.workers_approval)
        self.elites_approval = self._clamp_and_decay_approval(self.elites_approval)

        # 7. Elections (Every 4 years)
        if self.year % 4 == 0:
            average_approval = (self.farmers_approval + self.workers_approval + self.elites_approval) / 3.0
            # 45% threshold simulates a minority/coalition government
            if average_approval < 45.0:
                self.is_done = True
                self.game_over_reason = f"Lost election with {average_approval:.1f}% approval"
                return self._get_state(), self._calculate_reward(), self.is_done, {"reason": self.game_over_reason}

        # 8. End of Year Updates
        self.year += 1
        self.land_price = random.randint(17, 26)

        if self.year > self.max_years:
            self.is_done = True
            self.game_over_reason = "Completed term successfully!"

        return self._get_state(), self._calculate_reward(), self.is_done, {"reason": self.game_over_reason}

    def _clamp_and_decay_approval(self, approval):
        if approval > 50:
            approval -= (approval - 50) * 0.1
        elif approval < 50:
            approval += (50 - approval) * 0.1
        return max(0.0, min(100.0, approval))

    def _calculate_reward(self):
        survival_bonus = self.year * 100
        wealth_score = (self.land * 20 + self.grain) / 50.0
        pop_score = self.population * 2
        approval_score = min(self.farmers_approval, self.workers_approval, self.elites_approval) * 3.0
        penalty = self.starved_total * 50

        total_score = survival_bonus + wealth_score + pop_score + approval_score - penalty

        if self.is_done and self.year <= self.max_years:
            total_score = total_score / 10.0

        return total_score
