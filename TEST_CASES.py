"""Unit tests for the customer data processing script."""

import csv
import io
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import process_data
from process_data import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor using isolated temporary input and output files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.customers_file = os.path.join(self.temp_dir.name, "customers.csv")
        self.transactions_file = os.path.join(
            self.temp_dir.name, "transactions.csv"
        )
        self.customers = [
            {
                "customer_id": "C001",
                "name": "Alice Smith",
                "email": "alice@example.com",
                "join_date": "2024-01-10",
            },
            {
                "customer_id": "C002",
                "name": "Bob Jones",
                "email": "bob@example.com",
                "join_date": "2024-02-15",
            },
        ]
        self.transactions = [
            {
                "transaction_id": "T001",
                "customer_id": "C001",
                "amount": "10.00",
                "date": "2024-03-01",
                "category": "books",
            },
            {
                "transaction_id": "T002",
                "customer_id": "C002",
                "amount": "20.00",
                "date": "2024-03-02",
                "category": "games",
            },
            {
                "transaction_id": "T003",
                "customer_id": "C001",
                "amount": "30.00",
                "date": "2024-03-03",
                "category": "books",
            },
        ]
        self._write_csv(
            self.customers_file,
            ["customer_id", "name", "email", "join_date"],
            self.customers,
        )
        self._write_csv(
            self.transactions_file,
            ["transaction_id", "customer_id", "amount", "date", "category"],
            self.transactions,
        )
        self.processor = DataProcessor(self.customers_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_logging_writes_formatted_messages_to_both_destinations_once(self):
        """Logging writes all required levels to console and appended file once."""
        log_file = os.path.join(self.temp_dir.name, "test_error.log")
        with open(log_file, "w") as file:
            file.write("older log entry\n")

        terminal = io.StringIO()
        module_logger = process_data.logger
        original_handlers = module_logger.handlers[:]
        original_level = module_logger.level
        original_propagate = module_logger.propagate

        try:
            process_data.configure_logging(log_file, stream=terminal)
            module_logger.info("before reconfiguration")
            process_data.configure_logging(log_file, stream=terminal)

            configured_handlers = [
                handler
                for handler in module_logger.handlers
                if getattr(handler, "_process_data_handler", False)
            ]
            self.assertEqual(len(configured_handlers), 2)
            self.assertEqual(
                sum(
                    isinstance(handler, process_data.logging.StreamHandler)
                    and not isinstance(handler, process_data.logging.FileHandler)
                    for handler in configured_handlers),
                1,
            )
            self.assertEqual(
                sum(
                    isinstance(handler, process_data.logging.FileHandler)
                    for handler in configured_handlers),
                1,
            )

            module_logger.info("info message")
            module_logger.warning("warning message")
            module_logger.error("error message")

            for handler in configured_handlers:
                handler.flush()

            with open(log_file, "r") as file:
                file_lines = file.read().splitlines()
            terminal_lines = terminal.getvalue().splitlines()

            self.assertEqual(file_lines[0], "older log entry")
            self.assertEqual(file_lines[1:], terminal_lines)
            self.assertEqual(len(terminal_lines), 4)
            self.assertIn(" - INFO - info message", terminal_lines[-3])
            self.assertIn(" - WARNING - warning message", terminal_lines[-2])
            self.assertIn(" - ERROR - error message", terminal_lines[-1])
        finally:
            for handler in module_logger.handlers[:]:
                if getattr(handler, "_process_data_handler", False):
                    module_logger.removeHandler(handler)
                    handler.close()
            module_logger.handlers[:] = original_handlers
            module_logger.setLevel(original_level)
            module_logger.propagate = original_propagate

    @staticmethod
    def _write_csv(path, fieldnames, rows):
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_load_data_with_valid_customer_csv(self):
        """A valid customer CSV is loaded into the customer dictionary."""
        self.assertTrue(self.processor.load_data())

        self.assertEqual(len(self.processor.customers), 2)
        self.assertEqual(self.processor.customers["C001"]["name"], "Alice Smith")
        self.assertEqual(self.processor.customers["C001"]["total_spent"], 0.0)
        self.assertEqual(self.processor.customers["C001"]["transaction_count"], 0)

    def test_load_data_with_missing_file(self):
        """A missing customer file returns False instead of raising."""
        processor = DataProcessor(
            os.path.join(self.temp_dir.name, "missing_customers.csv")
        )

        self.assertFalse(processor.load_data())

    def test_load_data_with_invalid_csv_header(self):
        """A CSV missing a required column is rejected."""
        invalid_file = os.path.join(self.temp_dir.name, "invalid_customers.csv")
        self._write_csv(invalid_file, ["name"], [{"name": "Alice"}])
        processor = DataProcessor(invalid_file)

        self.assertFalse(processor.load_data())

    def test_process_transactions_with_valid_csv(self):
        """Valid transactions are stored and update known customers' totals."""
        self.assertTrue(self.processor.load_data())

        self.assertTrue(
            self.processor.process_transactions(self.transactions_file)
        )

        self.assertEqual(len(self.processor.transactions), 3)
        self.assertEqual(self.processor.customers["C001"]["total_spent"], 40.0)
        self.assertEqual(
            self.processor.customers["C001"]["transaction_count"], 2
        )
        self.assertEqual(self.processor.customers["C002"]["total_spent"], 20.0)

    def test_process_transactions_with_missing_file(self):
        """A missing transaction file returns False instead of raising."""
        missing_file = os.path.join(self.temp_dir.name, "missing_transactions.csv")

        self.assertFalse(self.processor.process_transactions(missing_file))

    def test_process_transactions_with_invalid_amount(self):
        """A non-numeric amount causes transaction processing to return False."""
        invalid_file = os.path.join(self.temp_dir.name, "invalid_transactions.csv")
        rows = [dict(self.transactions[0], amount="not-a-number")]
        self._write_csv(
            invalid_file,
            ["transaction_id", "customer_id", "amount", "date", "category"],
            rows,
        )

        self.assertFalse(self.processor.process_transactions(invalid_file))
        self.assertEqual(self.processor.transactions, [])

    def test_process_transactions_twice_skips_identical_transactions(self):
        """Repeating an identical transaction file does not change totals."""
        self.assertTrue(self.processor.load_data())
        self.assertTrue(
            self.processor.process_transactions(self.transactions_file)
        )
        original_transactions = list(self.processor.transactions)
        original_customers = {
            customer_id: dict(customer)
            for customer_id, customer in self.processor.customers.items()
        }

        with self.assertLogs("process_data", level="WARNING") as logs:
            self.assertTrue(
                self.processor.process_transactions(self.transactions_file)
            )

        self.assertEqual(self.processor.transactions, original_transactions)
        self.assertEqual(self.processor.customers, original_customers)
        self.assertTrue(any("Skipping duplicate" in message for message in logs.output))

    def test_identical_duplicate_rows_in_one_file_are_processed_once(self):
        """Identical duplicate IDs within one CSV are only counted once."""
        self.assertTrue(self.processor.load_data())
        duplicate_file = os.path.join(self.temp_dir.name, "duplicate_rows.csv")
        rows = [self.transactions[0], self.transactions[0]]
        self._write_csv(
            duplicate_file,
            ["transaction_id", "customer_id", "amount", "date", "category"],
            rows,
        )

        with self.assertLogs("process_data", level="WARNING") as logs:
            self.assertTrue(self.processor.process_transactions(duplicate_file))

        self.assertEqual(len(self.processor.transactions), 1)
        self.assertEqual(self.processor.customers["C001"]["total_spent"], 10.0)
        self.assertEqual(self.processor.customers["C001"]["transaction_count"], 1)
        self.assertTrue(any("Skipping duplicate" in message for message in logs.output))

    def test_process_transactions_conflicting_id_fails_without_changes(self):
        """A conflicting duplicate ID fails without applying any new rows."""
        self.assertTrue(self.processor.load_data())
        self.assertTrue(
            self.processor.process_transactions(self.transactions_file)
        )
        original_transactions = list(self.processor.transactions)
        original_customers = {
            customer_id: dict(customer)
            for customer_id, customer in self.processor.customers.items()
        }
        conflicting_file = os.path.join(self.temp_dir.name, "conflict.csv")
        conflicting_rows = [
            dict(self.transactions[0], transaction_id="T004"),
            dict(self.transactions[0], amount="99.00"),
        ]
        self._write_csv(
            conflicting_file,
            ["transaction_id", "customer_id", "amount", "date", "category"],
            conflicting_rows,
        )

        self.assertFalse(self.processor.process_transactions(conflicting_file))
        self.assertEqual(self.processor.transactions, original_transactions)
        self.assertEqual(self.processor.customers, original_customers)

    def test_process_transactions_invalid_later_row_is_atomic(self):
        """An invalid row after a valid one leaves processor state unchanged."""
        self.assertTrue(self.processor.load_data())
        invalid_file = os.path.join(self.temp_dir.name, "partially_invalid.csv")
        rows = [
            self.transactions[0],
            dict(self.transactions[1], amount="NaN"),
        ]
        self._write_csv(
            invalid_file,
            ["transaction_id", "customer_id", "amount", "date", "category"],
            rows,
        )

        self.assertFalse(self.processor.process_transactions(invalid_file))
        self.assertEqual(self.processor.transactions, [])
        self.assertEqual(self.processor.customers["C001"]["total_spent"], 0.0)
        self.assertEqual(self.processor.customers["C002"]["total_spent"], 0.0)

    def test_process_transactions_with_blank_required_value(self):
        """A blank required transaction field is rejected without state changes."""
        invalid_file = os.path.join(self.temp_dir.name, "blank_category.csv")
        rows = [dict(self.transactions[0], category=" ")]
        self._write_csv(
            invalid_file,
            ["transaction_id", "customer_id", "amount", "date", "category"],
            rows,
        )

        self.assertFalse(self.processor.process_transactions(invalid_file))
        self.assertEqual(self.processor.transactions, [])

    def test_load_data_invalid_row_is_atomic(self):
        """An invalid customer row does not leave earlier rows partially loaded."""
        invalid_file = os.path.join(self.temp_dir.name, "partially_invalid.csv")
        rows = [self.customers[0], dict(self.customers[1], email="")]
        self._write_csv(
            invalid_file,
            ["customer_id", "name", "email", "join_date"],
            rows,
        )
        processor = DataProcessor(invalid_file)

        self.assertFalse(processor.load_data())
        self.assertEqual(processor.customers, {})

    def test_process_transactions_conflict_after_new_row_is_atomic(self):
        """A later duplicate conflict prevents earlier new rows from committing."""
        self.assertTrue(self.processor.load_data())
        self.assertTrue(
            self.processor.process_transactions(self.transactions_file)
        )
        original_transactions = list(self.processor.transactions)
        original_customers = {
            customer_id: dict(customer)
            for customer_id, customer in self.processor.customers.items()
        }
        conflict_file = os.path.join(self.temp_dir.name, "late_conflict.csv")
        rows = [
            dict(self.transactions[0], transaction_id="T004"),
            dict(self.transactions[0], amount="99.00"),
        ]
        self._write_csv(
            conflict_file,
            ["transaction_id", "customer_id", "amount", "date", "category"],
            rows,
        )

        self.assertFalse(self.processor.process_transactions(conflict_file))
        self.assertEqual(self.processor.transactions, original_transactions)
        self.assertEqual(self.processor.customers, original_customers)

    def test_calculate_customer_metrics(self):
        """Metrics summarize the loaded customers and processed transactions."""
        self.assertTrue(self.processor.load_data())
        self.assertTrue(
            self.processor.process_transactions(self.transactions_file)
        )

        metrics = self.processor.calculate_customer_metrics()

        self.assertEqual(metrics["total_customers"], 2)
        self.assertEqual(metrics["total_transactions"], 3)
        self.assertEqual(metrics["total_revenue"], 60.0)
        self.assertEqual(metrics["average_transaction_value"], 20.0)
        self.assertEqual(metrics["category_breakdown"], {"books": 2, "games": 1})
        self.assertEqual(metrics["top_customers"][0][0], "C001")

    def test_metrics_keep_existing_behavior_for_unknown_customers(self):
        """Unknown-customer transactions are counted but add no customer revenue."""
        self.assertTrue(self.processor.load_data())
        unknown_file = os.path.join(self.temp_dir.name, "unknown_customer.csv")
        rows = [dict(self.transactions[0], customer_id="C999")]
        self._write_csv(
            unknown_file,
            ["transaction_id", "customer_id", "amount", "date", "category"],
            rows,
        )
        self.assertTrue(self.processor.process_transactions(unknown_file))

        metrics = self.processor.calculate_customer_metrics()

        self.assertEqual(metrics["total_transactions"], 1)
        self.assertEqual(metrics["total_revenue"], 0.0)
        self.assertEqual(metrics["average_transaction_value"], 0.0)
        self.assertEqual(metrics["category_breakdown"], {"books": 1})

    def test_find_matches(self):
        """Search is case-insensitive, matches substrings, and defaults to name."""
        self.assertTrue(self.processor.load_data())

        matches = self.processor.find_matches("ALICE")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["customer_id"], "C001")

        email_matches = self.processor.find_matches(
            "BOB@", field="email"
        )
        self.assertEqual([match["customer_id"] for match in email_matches], ["C002"])

    def test_generate_report_for_each_valid_type(self):
        """Each supported report type is written as readable JSON."""
        self.assertTrue(self.processor.load_data())
        self.assertTrue(
            self.processor.process_transactions(self.transactions_file)
        )
        expected_keys = {
            "customer_summary": "customers",
            "metrics": "metrics",
            "transactions": "transactions",
        }

        for report_type, payload_key in expected_keys.items():
            with self.subTest(report_type=report_type):
                output_file = os.path.join(
                    self.temp_dir.name, report_type + ".json"
                )

                self.assertTrue(
                    self.processor.generate_report(report_type, output_file)
                )
                with open(output_file, "r") as file:
                    report = json.load(file)

                self.assertIn("generated_at", report)
                self.assertIn(payload_key, report)

    def test_generate_report_for_invalid_type(self):
        """An unsupported report type returns False."""
        output_file = os.path.join(self.temp_dir.name, "unknown.json")

        self.assertFalse(
            self.processor.generate_report("unknown", output_file)
        )
        self.assertFalse(os.path.exists(output_file))

    def test_export_customer_data_as_csv(self):
        """CSV export includes customer IDs and the customer fields."""
        self.assertTrue(self.processor.load_data())
        output_file = os.path.join(self.temp_dir.name, "customers_export.csv")

        self.assertTrue(self.processor.export_customer_data(output_file, "csv"))

        with open(output_file, "r", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        self.assertIn("customer_id", reader.fieldnames)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["customer_id"], "C001")
        self.assertEqual(rows[0]["name"], "Alice Smith")

    def test_export_customer_data_csv_does_not_require_keys_method(self):
        """CSV export handles dict records whose keys method is unavailable."""
        self.assertTrue(self.processor.load_data())

        class CustomerRecord(dict):
            def keys(self):
                raise AttributeError("'dict' object has no attribute 'keys'")

        self.processor.customers["C001"] = CustomerRecord(
            self.processor.customers["C001"]
        )
        output_file = os.path.join(self.temp_dir.name, "customers_without_keys.csv")

        self.assertTrue(self.processor.export_customer_data(output_file, "csv"))

        with open(output_file, "r", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["customer_id"], "C001")
        self.assertEqual(rows[0]["name"], "Alice Smith")

    def test_export_customer_data_failure_is_logged_at_error(self):
        """A filesystem failure during export returns False and logs ERROR."""
        self.assertTrue(self.processor.load_data())
        output_file = os.path.join(
            self.temp_dir.name, "missing_directory", "customers.csv"
        )

        with self.assertLogs("process_data", level="ERROR") as logs:
            self.assertFalse(self.processor.export_customer_data(output_file, "csv"))

        self.assertTrue(any("Error exporting data" in message for message in logs.output))

    def test_export_customer_data_as_json(self):
        """JSON export stores customers under their customer IDs."""
        self.assertTrue(self.processor.load_data())
        output_file = os.path.join(self.temp_dir.name, "customers_export.json")

        self.assertTrue(self.processor.export_customer_data(output_file, "json"))

        with open(output_file, "r") as file:
            exported = json.load(file)

        self.assertEqual(exported["C001"]["name"], "Alice Smith")
        self.assertEqual(len(exported), 2)

    def test_export_customer_data_with_unsupported_format(self):
        """An unsupported export format returns False."""
        output_file = os.path.join(self.temp_dir.name, "customers_export.xml")

        self.assertFalse(self.processor.export_customer_data(output_file, "xml"))

    def test_main_reports_export_failure_and_attempts_all_outputs(self):
        """main logs failure instead of success and still tries every output."""
        processor = Mock()
        processor.load_data.return_value = True
        processor.process_transactions.return_value = True
        processor.generate_report.return_value = True
        processor.export_customer_data.side_effect = [False, True]

        with patch.object(process_data, "DataProcessor", return_value=processor):
            with self.assertLogs("process_data", level="ERROR") as logs:
                process_data.main()

        self.assertEqual(
            processor.generate_report.call_args_list,
            [
                call("customer_summary", "customer_summary.json"),
                call("metrics", "metrics.json"),
                call("transactions", "transactions.json"),
            ],
        )
        self.assertEqual(
            processor.export_customer_data.call_args_list,
            [
                call("customers_export.csv", "csv"),
                call("customers_export.json", "json"),
            ],
        )
        self.assertTrue(any("output failures" in message for message in logs.output))
        self.assertFalse(
            any("completed successfully" in message for message in logs.output)
        )

    def test_main_success_is_logged_at_info_not_error(self):
        """A successful pipeline completion is logged only at INFO level."""
        processor = Mock()
        processor.load_data.return_value = True
        processor.process_transactions.return_value = True
        processor.generate_report.return_value = True
        processor.export_customer_data.return_value = True

        with patch.object(process_data, "DataProcessor", return_value=processor):
            with self.assertLogs("process_data", level="INFO") as logs:
                process_data.main()

        completion_records = [
            record
            for record in logs.records
            if "Data processing completed successfully" in record.getMessage()
        ]
        self.assertEqual(len(completion_records), 1)
        self.assertEqual(completion_records[0].levelno, process_data.logging.INFO)
        self.assertFalse(
            any(record.levelno >= process_data.logging.ERROR for record in logs.records)
        )


if __name__ == "__main__":
    unittest.main()