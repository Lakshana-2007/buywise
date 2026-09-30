"""
Enumerations derived directly from the problem specification.
No magic strings anywhere else in the codebase — import from here.
"""

from enum import Enum


class AffordabilityStatus(str, Enum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class PaymentMethod(str, Enum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class RequestType(str, Enum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


class EventType(str, Enum):
    EXPENSE = "expense"
    INCOME = "income"
    DEBT_PAYMENT = "debt_payment"
    SUBSCRIPTION = "subscription"
    REFUND = "refund"
    INVESTMENT_PURCHASE = "investment_purchase"
    INVESTMENT_SALE = "investment_sale"
    INVESTMENT_VALUATION = "investment_valuation"


class EventStatus(str, Enum):
    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNREALIZED = "unrealized"


class EventDirection(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    NON_CASH = "non_cash"


class EventFlexibility(str, Enum):
    FIXED = "fixed"
    REDUCIBLE = "reducible"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"
    STOPPABLE = "stoppable"


class MessageSourceType(str, Enum):
    EMPLOYER = "employer"
    BANK = "bank"
    MERCHANT = "merchant"
    FINANCIAL_SERVICE = "financial_service"
    SERVICE_PROVIDER = "service_provider"


class Currency(str, Enum):
    INR = "INR"
    ZAR = "ZAR"
    IDR = "IDR"
    USD = "USD"
    EUR = "EUR"


# Events to IGNORE in cash-flow simulation
IGNORED_STATUSES: set[str] = {
    EventStatus.CANCELLED,
    EventStatus.FAILED,
    EventStatus.UNREALIZED,
}

# Pending credits must NOT be counted (spec: ignore pending credits)
# Pending debits ARE reserved (conservative)
NON_CASH_DIRECTIONS: set[str] = {EventDirection.NON_CASH}

# Event types that never contribute to spendable cash
NON_CASH_EVENT_TYPES: set[str] = {EventType.INVESTMENT_VALUATION}
