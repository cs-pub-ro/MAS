from openai import OpenAI

ACME_system_monotonic_concession = 'You are a company wishing to build a new headquarters.'\
+ 'You act as a contractor for four separate development stages: 1. structural design, 2. building, 3. plumbing and electrics, 4. interior design. ' \
+ 'The current stage is: "structural design". ' \
+ 'Following a reverse dutch auction you have selected the following companies: ["Company A", "Company C"].' \
+ 'The companies have responded to your offer of 4000 at round 2 out of 3.' \
+ 'You now start a monotonic concession negotiation with the selected companies. You want to start with a lower offer and increase it '\
+ 'back up to the budget for which the companies have bid in the reverse dutch auction.'\
+ 'There is a maximum of 3 rounds of offers.'\
+ 'You can increase the offer by how much you want for each round,until one or more companies make a counter-offer that is lower or equal to the one you made. ' \
+ 'You choose the company that offers the smallest counter-offer.'\
+ 'You must secure at least one company for this stage, otherwise you will lose the project.'\
+ 'You do not want to lose the project. '\
+ 'If you can, you will try to save as much money as possible.'

ACME_user_monotonic_concession = 'The negotation status is as follows:\n'\
+ '- current_stage: "structural design"\n'\
+ '- selected_companies: ["Company A", "Company C"]\n'\
+'- current_round: 2\n'\
+'- maximum_rounds: 3\n'\
+'- offer_after_dutch_auction: 4000\n'\
+'- number_of_companies: 2\n'\
+ '- previous_offer: 3600\n'\
+ '- counter_offers: {'\
+ '   "round": 1,'\
+ '   "offers": ['\
+ '        {'\
+ '         "company": "Company A",'\
+ '         "offer": 3900'\
+ '        },'\
+ '        {'\
+ '           "company": "Company C",'\
+ '           "offer": 3800'\
+ '       }'\
+ '     ]'\
+ ' }\n'\
+ 'Give the value for the `offer` for the current stage and the current monotonic concession round.'\
+ 'Think about it step by step. Analyze your risks and advantages.'\
+ 'Return the `offered_budget` in the following JSON dictionary format :\n'\
+ '```'\
+ '{'\
+ '    "stage": "structural design",'\
+ '    "offered_budget": <offered_budget>,'\
+ '    "round": <current_round>'\
+ '}'\
+ '```'


company_A_system_monotonic_concession = \
'You are a construction company called "Company A" participating in a project to build a new headquarters for a company called ACME.'\
+'There are four development stages: 1. structural design, 2. building, 3. plumbing and electrics, 4. interior design.'\
+'You specialize in the following stages: structural design, building, plumbing and electrics.'\
+'Your cost per stages in which you specialize is as follows:\n'\
+'```'\
+'cost_per_stage = {'\
+'    "structural design": 3602,'\
+'    "building": 11000,'\
+'    "plumbing and electrics": 3900'\
+'}'\
+'```'\
\
+'Following a reverse dutch auction you have been selected for the stages: ["structural design", "plumbing and electrics"].'\
+'Your goal is to secure at least one contract for your company. The overall revenue (over all stages for which you were selected) must be positive.'\
+'You cannot have overal negative revenue.'\
+'Winning a contract is more important than maximizing revenue.'\
+'If you can, you will try to win as much revenue as possible.'\
\
+'You now start a monotonic concession negotiation with ACME for the "plumbing and electrics" stage.'\
+'This means you are waiting for ACME to make an offer and you will respond with a counter-offer that must be lower or equal to the one you made in the previous round.'\
+'There are a maximum of 3 rounds of offers.'\
\
+'You know that ACME will choose the company that offers the smallest counter-offer.'\
+'You know that ACME must secure at least one company for this stage, otherwise they will lose the project.'\
+'You know that ACME does not want to lose the project.'

company_A_user_monotonic_concession = \
'You know that the total number of companies selected for the "plumbing and electrics" stage is: 2.'\
+'You know that your current number of won contracts is: 1.'\
+'You know that you have responded to a budget offer of 4500 in the reverse dutch auction.'\
+'Your know that your total expected revenue is: 110.'\
+'The negotation status is as follows:\n'\
+'  - current_stage: "plumbing and electrics"\n'\
+'  - number_of_stages_qualified_for: 2\n'\
+'  - number_of_stages_left: 1\n'\
+'  - number_of_won_contracts: 0\n'\
+'  - total_expected_revenue: 200\n'\
+'  - offer_after_dutch_auction: 4500\n'\
+'  - number_of_selected_companies: 2\n'\
+'  - current_negotiation_round: 3\n'\
+'  - maximum_negotiation_rounds: 3\n'\
+'  - received_offer: 4100\n'\
+'  - previous_counter_offer: 4400\n'\
+'Give the value for the `counter_offer` for the current stage and the current monotonic concession round.'\
+'Think about it step by step. Analyze your risks and advantages.'\
+'Return the `counter_offer` in the following JSON dictionary format:\n'\
+'```'\
+'{'\
+'    "stage": "structural design",'\
+'    "offer": <received_offer>,'\
+'    "counter_offer": <counter_offer>,'\
+'    "round": <current_round>'\
+'}'\
+'```'

if __name__ == '__main__':
    print("Running LLM Agent...")
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        seed=37,
        temperature=0.0,
        messages=[
            {
                "role": "system", 
                "content": 
                    "You are a company wishing to build a new headquarters" 
                  + "You act as a contractor for four separate development stages: 1. structural design, 2. building, 3. plumbing and electrics, 4. interior design."
                  + "You have six companies to choose from for each stage."
                  + "You start a reverse dutch auction for each stage, meaning that you start from a lower budget and increase it until  one or more companies accept the offer."
                  + "You have a maximum budget for each stage." 
                  + "You have at most 3 rounds of offers for each stage. You can increase the budget by how much you want for each round, but you cannot exceed the maximum budget per stage."
                  + "You must secure at least one company for each stage, otherwise you will lose the project."
                  + "You do not want to lose the project."
                  + "If you can, you will try to secure more than one company for each stage, in order to have a backup plan."
                  + "If you can, you will try to save as much money as possible."
            },
            {
                "role": "user", 
                "content": 
                  "Your budget for each stage is:\n" +
                  "```" +
                  "budget = {" +
                      "'structural design': 5000," +
                      "'building': 10000," +
                      "'plumbing and electrics': 4000," +
                      "'interior design': 5000" +
                  "}"+
                  "```"+
                  "The negotation status is as follows:\n" +
                    "- current_stage: 'structural design'\n" +
                    "- current_round: 3\n" +
                    "- maximum_rounds: 3\n" +
                    "- maximum_budget: 5000\n" +
                    "- number_of_companies: 6\n" +
                    "- companies: ['Company A', 'Company B', 'Company C', 'Company D', 'Company E', 'Company F']\n" +
                    "- previous_offer: 3750\n" +
                    "- accepting_companies: []\n" +

                  "Give the value for the `offered_budget` variable for the construction stage and the current round." +
                  "Think about it step by step. What would be the best strategy to secure at least one company for the current stage?" +
                  "Return the `offered_budget` in the following JSON dictionary format:\n" +
                  "```" +
                  "{" +
                      "'construction_stage': 'structural design'," +
                      "'offered_budget': <offered_budget>," +
                      "'round': <current_round>" +
                  "}"+
                  "```"
            },
        ]
    )
    
    print(response.choices[0].message.content)

    response_ACME = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        seed=37,
        temperature=0.0,
        messages=[
            {
                "role": "system", 
                "content": ACME_system_monotonic_concession
            },
            {
                "role": "user", 
                "content": ACME_user_monotonic_concession
            },
        ]
    )

    response_company_A = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        seed=37,
        temperature=0.0,
        messages=[
            {
                "role": "system", 
                "content": company_A_system_monotonic_concession
            },
            {
                "role": "user", 
                "content": company_A_user_monotonic_concession
            },
        ]
    )

    print("Response for ACME: ", response_ACME.choices[0].message.content)
    print("Response for Company A: ", response_company_A.choices[0].message.content)