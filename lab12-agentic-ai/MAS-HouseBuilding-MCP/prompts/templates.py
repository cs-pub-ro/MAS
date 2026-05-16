def format_auction_acme_prompt(
    task_name: str,
    budget: float,
    round: int,
    price: float,
    max_rounds: int,
    history: str,
) -> str:
    with open("prompts/auction/acme_auction_prompt.txt") as f:
        template = f.read()
    return template.format(
        task_name=task_name,
        budget=budget,
        round=round,
        price=price,
        max_rounds=max_rounds,
        history=history,
    )


def format_auction_company_prompt(
    name: str,
    task_name: str,
    price: float,
    cost: float,
    profit: float,
    specialties: str,
    history: str,
) -> str:
    with open("prompts/auction/company_auction_prompt.txt") as f:
        template = f.read()
    return template.format(
        name=name,
        task_name=task_name,
        price=price,
        cost=cost,
        profit=profit,
        specialties=specialties,
        history=history,
    )


def format_negotiation_acme_prompt(
    task_name: str,
    budget: float,
    bidders: str,
    offers: str,
    last_offers: str,
    recommended_action: str,
) -> str:
    with open("prompts/negotiation/acme_negotiation_prompt.txt") as f:
        template = f.read()
    return template.format(
        task_name=task_name,
        budget=budget,
        bidders=bidders,
        offers=offers,
        last_offers=last_offers,
        recommended_action=recommended_action,
    )


def format_negotiation_company_prompt(
    name: str,
    task_name: str,
    cost: float,
    budget: float,
    offers: str,
    last_offer: str,
    target_price: float,
    action: str,
) -> str:
    with open("prompts/negotiation/company_negotiation_prompt.txt") as f:
        template = f.read()
    return template.format(
        name=name,
        task_name=task_name,
        cost=cost,
        budget=budget,
        offers=offers,
        last_offer=last_offer,
        target_price=target_price,
        action=action,
    )
