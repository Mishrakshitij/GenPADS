"""Seven symbolic tasks with the slot/action counts from paper Table 2.

https://doi.org/10.1371/journal.pone.0278323.t002
The paper publishes task names and counts, but no complete action dictionaries.
These are new executable templates; only the flight slot names appear in its text.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    name: str
    template: str
    slot: str | None
    politeness: int

    @property
    def is_completion(self) -> bool:
        return self.slot is None


@dataclass(frozen=True)
class DomainSpec:
    name: str
    task: str
    slots: tuple[str, ...]
    actions: tuple[Action, ...]
    max_turns: int

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def state_dim(self) -> int:
        # Known slots, previous action, last user's four-class polite feedback.
        return len(self.slots) + self.num_actions + 4


def _domain(name, task, slots):
    actions = []
    for slot in slots:
        description = slot.replace("_", " ")
        actions.extend((
            Action(f"request_{slot}", f"What is your {description}?", slot, 0),
            Action(f"request_{slot}_polite",
                   f"Could you please share your {description}?", slot, 3),
        ))
    task_description = task.replace("_", " ")
    actions.extend((
        Action("complete", f"Your {task_description} is complete.", None, 1),
        Action("complete_polite",
               f"Thank you. Your {task_description} is complete. Happy to help!", None, 3),
    ))
    # The publication specifies 25–30 turns without per-domain limits.
    return DomainSpec(name, task, tuple(slots), tuple(actions), 30 if len(slots) == 5 else 25)


DOMAINS = {
    name: _domain(name, task, slots)
    for name, task, slots in (
        ("flights", "flight_search",
         ("round_trip_opt", "departure", "arrival", "dep_date", "dep_time")),
        ("food-ordering", "food_order",
         ("restaurant", "food_item", "quantity", "delivery_address", "delivery_time")),
        ("hotels", "hotel_search",
         ("destination", "check_in", "check_out", "guests", "price_range")),
        ("movies", "movie_search", ("movie_name", "theater", "date", "showtime")),
        ("music", "music_play", ("artist", "track", "album", "genre")),
        ("restaurant-search", "restaurant_search",
         ("location", "cuisine", "price_range", "party_size")),
        ("sports", "sports_team_search",
         ("sport", "league", "team", "location", "season")),
    )
}


def get_domain(domain: str | DomainSpec) -> DomainSpec:
    if isinstance(domain, DomainSpec):
        return domain
    try:
        return DOMAINS[domain]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown domain {domain!r}; choose from {', '.join(DOMAINS)}.") from exc
