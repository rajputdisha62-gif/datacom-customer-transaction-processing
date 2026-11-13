#!/usr/bin/env python3
"""
Data Processing Script for Customer Analytics
This script processes customer transaction data and generates reports.
"""

import csv
import heapq
import json
import logging
import math
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def configure_logging(log_file: str = "error.log", stream=None) -> None:
    """Configure one console and one append-mode file handler for this module."""
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        if getattr(handler, "_process_data_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler(stream)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

    for handler in (stream_handler, file_handler):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        handler._process_data_handler = True
        logger.addHandler(handler)


CUSTOMER_FIELDS = ("customer_id", "name", "email", "join_date")
TRANSACTION_FIELDS = (
    "transaction_id",
    "customer_id",
    "amount",
    "date",
    "category",
)


class DataProcessor:
    """Processes customer transaction data and generates analytics reports."""

    def __init__(self, input_file: str):
        """Initialize the data processor with input file path."""
        self.input_file = input_file
        self.customers = {}
        self.transactions = []
        self.reports = {}

    @staticmethod
    def _validate_headers(reader: csv.DictReader, required_fields, input_file: str):
        """Raise ValueError when a CSV lacks required headers."""
        if reader.fieldnames is None:
            raise ValueError(f"{input_file} is empty or has no header row")

        missing_fields = set(required_fields) - set(reader.fieldnames)
        if missing_fields:
            raise ValueError(
                f"{input_file} is missing required columns: "
                f"{', '.join(sorted(missing_fields))}"
            )

    @staticmethod
    def _required_value(row, field: str, input_file: str, row_number: int) -> str:
        """Return a required CSV value or raise a row-specific ValueError."""
        value = row.get(field)
        if value is None or not value.strip():
            raise ValueError(
                f"{input_file} row {row_number} has a missing {field}"
            )
        return value

    def load_data(self) -> bool:
        """Load customer and transaction data from CSV files."""
        try:
            loaded_customers = {}
            with open(self.input_file, "r", newline="") as file:
                reader = csv.DictReader(file)
                self._validate_headers(reader, CUSTOMER_FIELDS, self.input_file)

                for row_number, row in enumerate(reader, start=2):
                    customer_id = self._required_value(
                        row, "customer_id", self.input_file, row_number
                    )
                    loaded_customers[customer_id] = {
                        "name": self._required_value(
                            row, "name", self.input_file, row_number
                        ),
                        "email": self._required_value(
                            row, "email", self.input_file, row_number
                        ),
                        "join_date": self._required_value(
                            row, "join_date", self.input_file, row_number
                        ),
                        "total_spent": 0.0,
                        "transaction_count": 0,
                    }

            self.customers.update(loaded_customers)
            logger.info(f"Loaded {len(self.customers)} customers")
            return True
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            logger.error(
                f"Error loading customer data from {self.input_file}: {error}"
            )
            return False

    def process_transactions(self, transaction_file: str) -> bool:
        """Process transaction data and update customer records."""
        try:
            existing_transactions = {
                transaction["transaction_id"]: transaction
                for transaction in self.transactions
            }
            pending_transactions = {}

            with open(transaction_file, "r", newline="") as file:
                reader = csv.DictReader(file)
                self._validate_headers(reader, TRANSACTION_FIELDS, transaction_file)

                for row_number, row in enumerate(reader, start=2):
                    transaction_id = self._required_value(
                        row, "transaction_id", transaction_file, row_number
                    )
                    customer_id = self._required_value(
                        row, "customer_id", transaction_file, row_number
                    )
                    amount_text = self._required_value(
                        row, "amount", transaction_file, row_number
                    )
                    try:
                        amount = float(amount_text)
                    except ValueError as error:
                        raise ValueError(
                            f"{transaction_file} row {row_number} has an invalid amount"
                        ) from error
                    if not math.isfinite(amount):
                        raise ValueError(
                            f"{transaction_file} row {row_number} amount must be finite"
                        )

                    transaction = {
                        "transaction_id": transaction_id,
                        "customer_id": customer_id,
                        "amount": amount,
                        "date": self._required_value(
                            row, "date", transaction_file, row_number
                        ),
                        "category": self._required_value(
                            row, "category", transaction_file, row_number
                        ),
                    }

                    previous = pending_transactions.get(
                        transaction_id, existing_transactions.get(transaction_id)
                    )
                    if previous is not None:
                        if previous != transaction:
                            raise ValueError(
                                f"{transaction_file} row {row_number} conflicts "
                                f"with transaction ID {transaction_id}"
                            )
                        logger.warning(
                            f"Skipping duplicate transaction ID: {transaction_id}"
                        )
                        continue

                    pending_transactions[transaction_id] = transaction

            for transaction in pending_transactions.values():
                self.transactions.append(transaction)
                customer_id = transaction["customer_id"]
                if customer_id in self.customers:
                    self.customers[customer_id]["total_spent"] += transaction["amount"]
                    self.customers[customer_id]["transaction_count"] += 1
                else:
                    logger.warning(
                        f"Transaction for unknown customer: {customer_id}"
                    )

            logger.info(f"Processed {len(pending_transactions)} new transactions")
            return True
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            logger.error(
                f"Error processing transactions from {transaction_file}: {error}"
            )
            return False

    def calculate_customer_metrics(self) -> Dict[str, Any]:
        """Calculate various customer metrics and statistics."""
        if not self.customers:
            logger.error("No customer data available")
            return {}

        metrics = {
            "total_customers": len(self.customers),
            "total_transactions": len(self.transactions),
            "total_revenue": sum(
                cust["total_spent"] for cust in self.customers.values()
            ),
            "average_transaction_value": 0.0,
            "top_customers": [],
            "category_breakdown": {},
        }

        # Calculate average transaction value
        if metrics["total_transactions"] > 0:
            metrics["average_transaction_value"] = (
                metrics["total_revenue"] / metrics["total_transactions"]
            )

        # Find top customers by total spent
        metrics["top_customers"] = heapq.nlargest(
            10,
            self.customers.items(),
            key=lambda item: item[1]["total_spent"],
        )

        # Calculate category breakdown
        for transaction in self.transactions:
            category = transaction["category"]
            if category not in metrics["category_breakdown"]:
                metrics["category_breakdown"][category] = 0
            metrics["category_breakdown"][category] += 1

        return metrics

    def find_matches(
        self, search_term: str, field: str = "name"
    ) -> List[Dict[str, Any]]:
        """Find customers matching the search term in the specified field."""
        matches = []
        search_term_lower = search_term.lower()

        for customer_id, customer_data in self.customers.items():
            if field in customer_data:
                field_value = str(customer_data[field]).lower()
                if search_term_lower in field_value:
                    matches.append({"customer_id": customer_id, **customer_data})

        return matches

    def generate_report(self, report_type: str, output_file: str) -> bool:
        """Generate various types of reports and save to file."""
        try:
            if report_type == "customer_summary":
                report_data = {
                    "generated_at": datetime.now().isoformat(),
                    "customers": list(self.customers.values()),
                }
            elif report_type == "metrics":
                report_data = {
                    "generated_at": datetime.now().isoformat(),
                    "metrics": self.calculate_customer_metrics(),
                }
            elif report_type == "transactions":
                report_data = {
                    "generated_at": datetime.now().isoformat(),
                    "transactions": self.transactions,
                }
            else:
                logger.error(f"Unknown report type: {report_type}")
                return False

            # Save report to file
            with open(output_file, "w") as file:
                json.dump(report_data, file, indent=2)

            logger.info(f"Generated {report_type} report: {output_file}")
            return True

        except (OSError, UnicodeError, TypeError, ValueError) as error:
            logger.error(
                f"Error generating {report_type} report at {output_file}: {error}"
            )
            return False

    def export_customer_data(self, output_file: str, format: str = "csv") -> bool:
        """Export customer data in specified format."""
        try:
            if format == "csv":
                with open(output_file, "w", newline="") as file:
                    if self.customers:
                        first_customer = next(iter(self.customers.values()))
                        fieldnames = ["customer_id"] + list(first_customer)
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        writer.writeheader()

                        for customer_id, data in self.customers.items():
                            row = {"customer_id": customer_id}
                            for field in data:
                                row[field] = data[field]
                            writer.writerow(row)
            elif format == "json":
                with open(output_file, "w") as file:
                    json.dump(self.customers, file, indent=2)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

            logger.info(f"Exported customer data to {output_file}")
            return True

        except (OSError, UnicodeError, csv.Error, TypeError, ValueError) as error:
            logger.error(f"Error exporting data to {output_file}: {error}")
            return False


def main():
    """Main function to run the data processing pipeline."""
    # Initialize processor
    processor = DataProcessor("customers.csv")

    # Load data
    if not processor.load_data():
        logger.error("Failed to load customer data")
        return

    # Process transactions
    if not processor.process_transactions("transactions.csv"):
        logger.error("Failed to process transactions")
        return

    outputs_succeeded = True
    for report_type, output_file in (
        ("customer_summary", "customer_summary.json"),
        ("metrics", "metrics.json"),
        ("transactions", "transactions.json"),
    ):
        if not processor.generate_report(report_type, output_file):
            logger.error(f"Failed to generate {report_type} report")
            outputs_succeeded = False

    for output_file, export_format in (
        ("customers_export.csv", "csv"),
        ("customers_export.json", "json"),
    ):
        if not processor.export_customer_data(output_file, export_format):
            logger.error(f"Failed to export customer data as {export_format}")
            outputs_succeeded = False

    if outputs_succeeded:
        logger.info("Data processing completed successfully")
    else:
        logger.error("Data processing completed with output failures")


if __name__ == "__main__":
    configure_logging()
    main()
