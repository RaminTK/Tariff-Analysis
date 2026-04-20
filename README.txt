# Tariff misinformation supply chain model

Files:
- gurobi_tariff_misinfo_model.py: two-stage stochastic Gurobi model
- scenario_generator.py: Bayesian noisy-signal generator for tariff and misinformation scenarios
- dataset_template.xlsx: input workbook template

Recommended workflow:
1. Fill core network and cost sheets in dataset_template.xlsx
2. Fill TariffPriors and SignalDesign
3. Run:
   python scenario_generator.py dataset_template.xlsx
4. Run:
   python gurobi_tariff_misinfo_model.py dataset_template.xlsx

Main required generated model sheets:
- Signals
- TariffScenarios
- ConditionalProb

Core required sheets:
- Suppliers
- Warehouses
- DemandZones
- InboundCosts
- OutboundCosts

Notes:
- perceived tariff = scenario-based planning input inferred from signal states
- actual tariff = realized tariff scenario used for ex post evaluation
