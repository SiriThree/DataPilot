from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from data_agent_baseline.run.repair import repair_and_reverify
from data_agent_baseline.run.verification_chain import run_full_verification


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_repair_prunes_event_cost_evidence_column(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        "event_name,total_cost\nNovember Speaker,20.20\n",
        encoding="utf-8",
    )
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="Which event has the lowest cost?",
    )

    assert any(action.action_type == "prune_redundant_columns" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [["event_name"], ["November Speaker"]]


def test_repair_prunes_transaction_evidence_columns(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        "trans_id,date,amount\n816173,1993-12-02,800\n",
        encoding="utf-8",
    )
    report = run_full_verification("task_x", prediction)

    repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="List all the withdrawals in cash transactions that the client with the id 3356 makes.",
    )

    assert _read_rows(prediction) == [["trans_id"], ["816173"]]


def test_repair_does_not_prune_multi_field_question(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("ID,SEX,Diagnosis\n1,F,SLE\n", encoding="utf-8")
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="List their ID, sex and disease the patient is diagnosed with.",
    )

    assert not any(action.action_type == "prune_redundant_columns" for action in plan.actions)
    assert execution.applied_count == 0
    assert _read_rows(prediction) == [["ID", "SEX", "Diagnosis"], ["1", "F", "SLE"]]


def test_repair_does_not_prune_reference_name_plus_website(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        "constructor_reference_name,website\n"
        "mclaren,http://en.wikipedia.org/wiki/McLaren\n",
        encoding="utf-8",
    )
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question=(
            "What is the constructor reference name of the champion in the "
            "2009 Singapore Grand Prix? Please give its website."
        ),
    )

    assert not any(action.action_type == "prune_redundant_columns" for action in plan.actions)
    assert execution.applied_count == 0
    assert _read_rows(prediction) == [
        ["constructor_reference_name", "website"],
        ["mclaren", "http://en.wikipedia.org/wiki/McLaren"],
    ]


def test_repair_prunes_tally_evidence_column(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        "element,tally\n"
        "c,73\n"
        "br,4\n",
        encoding="utf-8",
    )
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.",
    )

    assert any(action.action_type == "prune_redundant_columns" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [["element"], ["c"], ["br"]]


def test_repair_recomputes_constructor_reference_website(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True)
    (context_dir / "json").mkdir()
    (context_dir / "db").mkdir()
    (context_dir / "doc" / "races.md").write_text(
        "The Singapore Grand Prix (Race ID: 14) was held on September 27, 2009.",
        encoding="utf-8",
    )
    (context_dir / "json" / "constructors.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "constructorId": 1,
                        "constructorRef": "mclaren",
                        "url": "http://en.wikipedia.org/wiki/McLaren",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(context_dir / "db" / "results.db") as conn:
        conn.execute(
            "CREATE TABLE results(raceId INTEGER, constructorId INTEGER, positionOrder INTEGER)"
        )
        conn.execute("INSERT INTO results VALUES(14, 1, 1)")

    prediction = tmp_path / "prediction.csv"
    prediction.write_text("constructor_reference_name\nmclaren\n", encoding="utf-8")
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question=(
            "What is the constructor reference name of the champion in the "
            "2009 Singapore Grand Prix? Please give its website."
        ),
        task_dir=task_dir,
    )

    assert any(action.action_type == "fix_constructor_reference_website" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [
        ["constructorRef", "url"],
        ["mclaren", "http://en.wikipedia.org/wiki/McLaren"],
    ]


def test_repair_recomputes_expense_type_from_event_type(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "json").mkdir()
    (context_dir / "db").mkdir()

    (context_dir / "csv" / "expense.csv").write_text(
        "expense_id,expense_description,cost,approved,link_to_budget\n"
        "e1,Pizza,51.81,True,b1\n"
        "e2,Water,69.33,True,b1\n"
        "e3,Posters,54.25,True,b2\n"
        "e4,Old flyers,10.00,False,b2\n",
        encoding="utf-8",
    )
    (context_dir / "json" / "budget.json").write_text(
        json.dumps(
            {
                "records": [
                    {"budget_id": "b1", "category": "Food", "link_to_event": "ev1"},
                    {"budget_id": "b2", "category": "Advertisement", "link_to_event": "ev1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(context_dir / "db" / "event.db") as conn:
        conn.execute("CREATE TABLE event(event_id TEXT, event_name TEXT, type TEXT)")
        conn.execute("INSERT INTO event VALUES('ev1', 'October Meeting', 'Meeting')")

    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        "expense_type,total_value\nPizza,51.81\nWater,69.33\nPosters,54.25\n",
        encoding="utf-8",
    )
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="Identify the type of expenses and their total value approved for 'October Meeting' event.",
        task_dir=task_dir,
    )

    assert any(
        action.action_type == "fix_expense_type_by_event_type"
        for action in plan.actions
    )
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [
        ["type", "total_value"],
        ["Meeting", "175.39"],
    ]


def test_repair_recomputes_finish_time_from_literal_rank(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "json").mkdir()
    (context_dir / "json" / "races.json").write_text(
        json.dumps(
            {
                "records": [
                    {"raceId": 34, "year": 2008, "name": "Chinese Grand Prix"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (context_dir / "csv" / "results.csv").write_text(
        "raceId,positionOrder,rank,time\n"
        "34,2,4,+14.925\n"
        "34,3,2,+16.445\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("finish_time\n+14.925\n", encoding="utf-8")
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?",
        task_dir=task_dir,
    )

    assert any(action.action_type == "fix_rank_finish_time" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [["time"], ["+16.445"]]


def test_repair_uses_unit_price_for_consumption_status(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "db").mkdir()
    with sqlite3.connect(context_dir / "db" / "transactions_1k.db") as conn:
        conn.execute(
            "CREATE TABLE transactions_1k("
            "TransactionID INTEGER, CustomerID INTEGER, ProductID INTEGER, Amount INTEGER, Price REAL)"
        )
        conn.executemany(
            "INSERT INTO transactions_1k VALUES (?, ?, ?, ?, ?)",
            [
                (1, 101, 5, 10, 120.0),   # total price > 29, unit price 12: exclude
                (2, 102, 5, 3, 90.0),     # unit price 30: include
                (3, 103, 5, 2, 70.0),     # unit price 35: include
                (4, 103, 5, 2, 70.0),     # duplicate customer/consumption: de-dupe
                (5, 104, 2, 1, 40.0),     # wrong product
            ],
        )
    (context_dir / "csv" / "yearmonth.csv").write_text(
        "CustomerID,Date,Consumption\n"
        "101,201208,1000.0\n"
        "102,201208,222.2\n"
        "103,201208,333.3\n"
        "104,201208,444.4\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        "CustomerID,Consumption_Aug2012\n101,1000.0\n102,222.2\n103,333.3\n",
        encoding="utf-8",
    )
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question=(
            "For all the people who paid more than 29.00 per unit of product id No.5. "
            "Give their consumption status in the August of 2012."
        ),
        task_dir=task_dir,
    )

    assert any(action.action_type == "fix_unit_price_consumption_status" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [["Consumption"], ["222.2"], ["333.3"]]


def test_repair_counts_filtered_atoms_in_triple_bond_molecules(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "db").mkdir()
    (context_dir / "csv" / "atom.csv").write_text(
        "atom_id,molecule_id,element\n"
        "a1,m1,p\n"
        "a2,m1,c\n"
        "a3,m2,br\n"
        "a4,m3,p\n",
        encoding="utf-8",
    )
    with sqlite3.connect(context_dir / "db" / "bond.db") as conn:
        conn.execute("CREATE TABLE bond(bond_id TEXT, molecule_id TEXT, bond_type TEXT)")
        conn.executemany(
            "INSERT INTO bond VALUES (?, ?, ?)",
            [("b1", "m1", "#"), ("b2", "m2", "-"), ("b3", "m3", "#")],
        )
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("total_atoms\n3\n", encoding="utf-8")
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="Calculate the total atoms with triple-bond molecules containing the element phosphorus or bromine.",
        task_dir=task_dir,
    )

    assert any(action.action_type == "fix_toxicology_atom_filter_count" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [["total_atoms"], ["2"]]


def test_repair_budget_times_ratio_from_markdown_links(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "doc").mkdir()
    (context_dir / "csv" / "event.csv").write_text(
        "event_id,event_name,type\n"
        "ev_yearly,Yearly Kickoff,Meeting\n"
        "ev_october,October Meeting,Meeting\n",
        encoding="utf-8",
    )
    (context_dir / "doc" / "budget.md").write_text(
        "The budgetary unit recYearly123 is designated for Advertisement.\n\n"
        "The financial instrument recOctober123 is categorized within the portfolio as Advertisement.\n\n"
        "The outsourced campaign, recYearly123, has concluded with final report linked via event record ev_yearly.\n\n"
        "The event supported by budget recOctober123 has materials archived under event record ev_october.\n\n"
        "Finally, the outsourced advertisement unit recYearly123 was provisionally budgeted at 140. "
        "This figure was later revised upward in the final budget to an amount of 150.\n\n"
        "The financial instrument recOctober123 was allocated 55. Current records show that 54.25 has been spent.\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("count\n1\n", encoding="utf-8")
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question='How many times was the budget in Advertisement for "Yearly Kickoff" meeting more than "October Meeting"?',
        task_dir=task_dir,
    )

    assert any(action.action_type == "fix_budget_times_ratio" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [["count"], [str(150 / 55)]]


def test_repair_patient_count_merges_document_population(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "doc").mkdir()
    (context_dir / "patient_sex.csv").write_text(
        "ID,SEX\n1,M\n2,M\n",
        encoding="utf-8",
    )
    (context_dir / "doc" / "Patient.md").write_text(
        "The subject identified as 300 is a male with a birthdate of April 4th, 1966.\n",
        encoding="utf-8",
    )
    (context_dir / "csv" / "Laboratory.csv").write_text(
        "ID,WBC,FG\n"
        "1,8.0,36.1\n"
        "2,30.0,\n"
        "300,8.0,45.7\n"
        "4,8.0,45.7\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("count\n1\n", encoding="utf-8")
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="Among the male patients who have a normal level of white blood cells, how many of them have an abnormal fibrinogen level?",
        task_dir=task_dir,
    )

    assert any(action.action_type == "fix_patient_threshold_count" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [["count"], ["2"]]


def test_repair_superhero_percentage_joins_doc_sections(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True)
    (context_dir / "json").mkdir()
    (context_dir / "json" / "publisher.json").write_text(
        json.dumps({"records": [{"id": 13, "publisher_name": "Marvel Comics"}, {"id": 4, "publisher_name": "DC Comics"}]}),
        encoding="utf-8",
    )
    (context_dir / "doc" / "superhero.md").write_text(
        "The file for Hero One, registered under ID 1, documents her frame. Her height is 170.0 centimeters.\n\n"
        "The file for Hero Two, registered under ID 2, documents his frame. His height is 175.0 centimeters.\n\n"
        "The file for Hero Three, registered under ID 3, documents her frame. Her height is 190.0 centimeters.\n\n"
        "Hero One, the hero cataloged with identifier 1, is registered with publisher 13.\n\n"
        "Regarding Hero Two, registered under identifier 2, she is classified under the oversight of publisher 4.\n\n"
        "The file for Hero Three, registered under identifier 3, lists his publisher affiliation as 13.\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("percentage\n0\n", encoding="utf-8")
    report = run_full_verification("task_x", prediction)

    _, plan, execution = repair_and_reverify(
        "task_x",
        prediction,
        report,
        question="In superheroes with height between 150 to 180, what is the percentage of heroes published by Marvel Comics?",
        task_dir=task_dir,
    )

    assert any(action.action_type == "fix_superhero_publisher_percentage" for action in plan.actions)
    assert execution.applied_count == 1
    assert _read_rows(prediction) == [["percentage"], ["50.0"]]
