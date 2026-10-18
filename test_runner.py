"""
Granularer ML-Statistik-Kurs Test Runner
========================================
Testet jede Funktion einzeln! Du musst NICHT alle Funktionen einer Übung
fertigstellen, um deine ersten Erfolge zu sehen und Teilpunkte zu prüfen.

Verwendung:
  py test_runner.py                    # Zeigt den Status ALLER einzelnen Funktionen
  py test_runner.py --module 1         # Testet alle Funktionen von Modul 1 einzeln
  py test_runner.py -m 1               # Kurzform für Modul 1
  py test_runner.py --function mean    # Testet gezielt eine bestimmte Funktion (auch Teilnamen möglich)
  py test_runner.py -f mean            # Kurzform für gezielte Funktionssuche
  py test_runner.py --use-solutions    # Verifiziert die Musterlösungen
"""

import argparse
import importlib
import os
import sys
import traceback

# Pfade konfigurieren
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tests.test_all import MODULE_FUNCTION_TESTS

MODULE_METADATA = [
    {
        "id": 1,
        "name": "Deskriptive Statistik & EDA",
        "ex_file": "exercises.ex01_descriptive_stats",
        "sol_file": "solutions.sol01_descriptive_stats",
    },
    {
        "id": 2,
        "name": "Wahrscheinlichkeit & Bayes-Theorem",
        "ex_file": "exercises.ex02_probability_bayes",
        "sol_file": "solutions.sol02_probability_bayes",
    },
    {
        "id": 3,
        "name": "Verteilungen & Zentraler Grenzwertsatz",
        "ex_file": "exercises.ex03_distributions_clt",
        "sol_file": "solutions.sol03_distributions_clt",
    },
    {
        "id": 4,
        "name": "Hypothesentests & A/B-Testing",
        "ex_file": "exercises.ex04_hypothesis_ab_testing",
        "sol_file": "solutions.sol04_hypothesis_ab_testing",
    },
    {
        "id": 5,
        "name": "MLE & Loss-Funktionen (MSE / Cross-Entropy)",
        "ex_file": "exercises.ex05_mle_and_loss_functions",
        "sol_file": "solutions.sol05_mle_and_loss_functions",
    },
    {
        "id": 6,
        "name": "Informationstheorie (Entropie & KL-Divergenz)",
        "ex_file": "exercises.ex06_information_theory",
        "sol_file": "solutions.sol06_information_theory",
    },
    {
        "id": 7,
        "name": "Statistische ML-Evaluation (ROC-AUC, F1, Brier)",
        "ex_file": "exercises.ex07_ml_evaluation_metrics",
        "sol_file": "solutions.sol07_ml_evaluation_metrics",
    },
    {
        "id": 8,
        "name": "Bootstrapping & Ensemble-Statistik (Bagging)",
        "ex_file": "exercises.ex08_bootstrapping_resampling",
        "sol_file": "solutions.sol08_bootstrapping_resampling",
    },
    {
        "id": 9,
        "name": "Multivariate Statistik & PCA",
        "ex_file": "exercises.ex09_multivariate_pca",
        "sol_file": "solutions.sol09_multivariate_pca",
    },
    {
        "id": 10,
        "name": "Data Drift & Monitoring in MLOps (PSI, KS)",
        "ex_file": "exercises.ex10_distribution_drift_monitoring",
        "sol_file": "solutions.sol10_distribution_drift_monitoring",
    },
]


def test_single_function(mod, func_name: str, test_func) -> tuple:
    """
    Testet eine einzelne Funktion isoliert.
    Rückgabe: (status_code, nachricht)
    status_code: 'PASS', 'TODO', 'FAIL', 'MISSING'
    """
    if not hasattr(mod, func_name):
        return "MISSING", f"Funktion '{func_name}' existiert nicht in der Datei."

    try:
        test_func(mod)
        return "PASS", "Erfolgreich validiert."
    except NotImplementedError as e:
        msg = str(e) or "Noch nicht implementiert (NotImplementedError)"
        return "TODO", msg
    except AssertionError as e:
        return "FAIL", f"Falsches Ergebnis (AssertionError: {e})"
    except Exception as e:
        return "FAIL", f"Laufzeitfehler ({type(e).__name__}: {e})"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Granularer Test-Runner: Teste jede Statistik-Funktion einzeln!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Beispiele:
  py test_runner.py -m 1                  # Testet alle Einzelfunktionen von Modul 1
  py test_runner.py -f calculate_mean     # Testet nur 'calculate_mean'
  py test_runner.py -f bayes              # Testet alle Funktionen mit 'bayes' im Namen
"""
    )
    parser.add_argument("--module", "-m", type=int, choices=range(1, 11), help="Nummer der Übung (1-10)")
    parser.add_argument("--function", "-f", type=str, help="Name oder Teilname einer Funktion zum gezielten Einzeltest")
    parser.add_argument("--use-solutions", action="store_true", help="Offizielle Musterlösungen testen")
    parser.add_argument("--verbose", "-v", action="store_true", help="Ausführliche Fehlermeldungen bei FAIL anzeigen")
    args = parser.parse_args()

    mode_label = "LOESUNGEN (solutions/)" if args.use_solutions else "DEIN CODE (exercises/)"
    print("=" * 72)
    print(">> STATISTIK FUER AI/ML-ENGINEERS: GRANULARER TEST-RUNNER")
    print(f">> Modus: {mode_label}")
    if args.module:
        print(f">> Filter: Nur Modul {args.module}")
    if args.function:
        print(f">> Filter: Nur Funktionen mit Suchbegriff '{args.function}'")
    print("=" * 72)

    total_funcs = 0
    passed_funcs = 0
    todo_funcs = 0
    failed_funcs = 0

    target_modules = [m for m in MODULE_METADATA if args.module is None or m["id"] == args.module]

    for mod_meta in target_modules:
        mod_id = mod_meta["id"]
        mod_name = mod_meta["name"]
        mod_import_path = mod_meta["sol_file"] if args.use_solutions else mod_meta["ex_file"]

        # Filtere Tests für dieses Modul
        func_tests = MODULE_FUNCTION_TESTS.get(mod_id, [])
        if args.function:
            func_tests = [ft for ft in func_tests if args.function.lower() in ft[0].lower()]

        if not func_tests:
            # Falls kein Filter zutraf
            continue

        # Modul laden
        try:
            mod = importlib.import_module(mod_import_path)
            mod = importlib.reload(mod)
            load_error = None
        except Exception as e:
            mod = None
            load_error = f"Datei konnte nicht geladen werden ({type(e).__name__}: {e})"

        print(f"\n📁 Modul {mod_id:02d}: {mod_name}")
        print(f"   Datei: {mod_import_path.replace('.', '/')}.py")
        print("   " + "-" * 66)

        if load_error:
            print(f"   [LOAD-ERROR] {load_error}")
            failed_funcs += len(func_tests)
            total_funcs += len(func_tests)
            continue

        mod_passed = 0
        mod_total = len(func_tests)

        for func_name, test_fn in func_tests:
            total_funcs += 1
            status, detail = test_single_function(mod, func_name, test_fn)

            if status == "PASS":
                icon = "[PASS]"
                passed_funcs += 1
                mod_passed += 1
                print(f"   {icon:<7} {func_name:<34} (Bestanden)")
            elif status == "TODO":
                icon = "[TODO]"
                todo_funcs += 1
                print(f"   {icon:<7} {func_name:<34} (Noch nicht geloest)")
            elif status == "MISSING":
                icon = "[MISS]"
                failed_funcs += 1
                print(f"   {icon:<7} {func_name:<34} (Funktion fehlt im Code)")
            else:  # FAIL
                icon = "[FAIL]"
                failed_funcs += 1
                print(f"   {icon:<7} {func_name:<34} (Fehlgeschlagen)")
                print(f"           -> {detail}")

        pct = (mod_passed / mod_total) * 100 if mod_total > 0 else 0
        print(f"   -> Status Modul {mod_id}: {mod_passed}/{mod_total} Funktionen bestanden ({pct:.0f}%)")

    print("\n" + "=" * 72)
    print(f"GESAMT-ERGEBNIS: {passed_funcs} von {total_funcs} Einzelfunktionen erfolgreich geloest!")
    print(f"Details: {passed_funcs} [PASS] | {todo_funcs} [TODO] | {failed_funcs} [FAIL/MISSING]")
    
    if passed_funcs == total_funcs and total_funcs > 0:
        print("Perfekt! Alle geprueften Funktionen funktionieren einwandfrei!")
    elif passed_funcs > 0:
        print(f"Klasse Fortschritt! Du hast bereits {passed_funcs} Funktionen gemeistert.")
        print("Arbeite Schritt fuer Schritt an den naechsten [TODO]s weiter.")
    else:
        print("Tipp: Du kannst jede Funktion einzeln lösen und sofort testen.")
        print("Beispiel: py test_runner.py -f calculate_mean")
    print("=" * 72)


if __name__ == "__main__":
    main()
