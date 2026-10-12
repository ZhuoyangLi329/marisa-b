#!/usr/bin/env python3
"""Generate the audited real-space B411 FFTLog and finite-UV tables.

The upstream WDX files are from oliverphilcox/OneLoopBispectrum (GPL-3.0).
This decoder intentionally supports only the dense scalar-array subset used
by tab411freq-flat.wdx and tab411derivs-flat.wdx, plus the single compressed
FullForm expression stored in b411uv-flat.wdx.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Union


EXPECTED_FREQUENCY_SHA256 = (
    "8a3cf8f0853504c06c705bc8598e0ab71278458693b02d777f9e4031bdd09f33"
)
EXPECTED_COEFFICIENT_SHA256 = (
    "df91008f35ef12ffb7163716e7897b449264e2f9ddb52c57a70413964639a059"
)
EXPECTED_UV_SHA256 = (
    "5ed29ddf0b6483116c44876771297e470f64413c1ad9225f545706fae15db291"
)
EXPECTED_B222_FREQUENCY_SHA256 = (
    "803531c5786dd5b4416d277a3abe74f57204444902949b42bf82b54fe8125fd1"
)
EXPECTED_B222_COEFFICIENT_SHA256 = (
    "5eb75bd7033d2786fb885ff448215ca6f1b8b00e4084aeb43552eec89b3cba7b"
)
EXPECTED_B321I_FREQUENCY_SHA256 = (
    "9d348de88bb263e63d695d1d29c3c9141373e6a27b91973b00ad6e5768e86171"
)
EXPECTED_B321I_COEFFICIENT_SHA256 = (
    "99ad3047700fea373762c95b008ea2ece1e670869da37e3c3fb8e377f54bb280"
)

# Actual Mathematica Union ordering, also recorded verbatim by the upstream
# "Bispectrum Interpolation.ipynb".  b3 is retained even though its entire
# opposite-pair B411 row is analytically zero.
BIAS_ROWS = (
    ("B1", 0),
    ("B2", 1),
    ("B3", 2),
    ("Gamma2", 31),
    ("Gamma21", 40),
    ("Gamma211", 46),
    ("Gamma21x", 49),
    ("Gamma22", 52),
    ("Gamma2x", 55),
    ("Gamma3", 61),
    ("Gamma31", 67),
)

# Rows in the upstream Mathematica ``Union`` ordering which survive after
# setting the redshift-space growth-rate symbol f to zero.  Their order is
# the stable operator order used by the generated C++ tables.
B222_REALSPACE_ROWS = (0, 1, 2, 3, 30, 31, 32, 44, 45, 49)
B321I_REALSPACE_ROWS = (
    0, 1, 2, 3, 4, 37, 38, 39, 52,
    56, 57, 64, 66, 67, 74, 76, 77, 84,
)

Scalar = Union[int, Fraction, str]
Polynomial = dict[tuple[int, int], Fraction]

UV_BIAS_INDEX = {
    "b1": 0,
    "b2": 1,
    "b3": 2,
    "g2": 3,
    "g21": 4,
    "g211": 5,
    "g21x": 6,
    "g22": 7,
    "g2x": 8,
    "g3": 9,
    "g31": 10,
}


class WdxDenseArray:
    """Minimal WDX1 dense-array decoder for the two audited upstream files."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        if self.data[:4] != b"WDX1":
            raise ValueError(f"{path} is not a WDX1 file")
        if self.data[0x20:0x22] != b"\xa1\x02":
            raise ValueError(f"{path} does not contain the expected rank-2 array")
        self.rows = self.data[0x22]
        self.columns = self.data[0x23]
        self.offset = 0x24

    def _compressed_expression(self) -> str:
        if self.data[self.offset] != 0xE0:
            raise ValueError("internal WDX compressed-expression parse error")
        if self.data[self.offset + 1] == 0xFF:
            length = int.from_bytes(
                self.data[self.offset + 2 : self.offset + 10], "little"
            )
            start = self.offset + 10
        else:
            length = self.data[self.offset + 1]
            start = self.offset + 2
        stop = start + length
        value = zlib.decompress(self.data[start:stop]).decode("utf-8")
        self.offset = stop
        return value

    @staticmethod
    def _wrapped_rational_integer(value: Scalar, head: str) -> int:
        if not isinstance(value, str):
            raise ValueError("WDX rational child is not an expression")
        match = re.fullmatch(
            rf"{head}\[Rational\[(-?\d+), (\d+)\]\]", value
        )
        if match is None:
            raise ValueError(f"unexpected WDX rational child: {value}")
        numerator, denominator = map(int, match.groups())
        return numerator if head == "Numerator" else denominator

    def scalar(self) -> Scalar:
        token = self.data[self.offset]
        if token == 0x02:
            # Compact WDX symbol.  The public B222 table contains one bare
            # ``x`` monomial encoded this way instead of as a compressed
            # FullForm expression.
            length = self.data[self.offset + 1]
            start = self.offset + 2
            stop = start + length
            value = self.data[start:stop].decode("utf-8")
            self.offset = stop
            return value
        if token == 0x10:
            value = struct.unpack(
                "<i", self.data[self.offset + 1 : self.offset + 5]
            )[0]
            self.offset += 5
            return value
        if token == 0x23:
            # WDX rational constructor: the following two scalar records are
            # Numerator[Rational[...]] and Denominator[Rational[...]].
            self.offset += 1
            numerator = self._wrapped_rational_integer(
                self.scalar(), "Numerator"
            )
            denominator = self._wrapped_rational_integer(
                self.scalar(), "Denominator"
            )
            return Fraction(numerator, denominator)
        if token == 0xE0:
            return self._compressed_expression()
        raise ValueError(
            f"unsupported WDX token 0x{token:02x} at 0x{self.offset:x}"
        )

    def decode(self) -> list[list[Scalar]]:
        values = [self.scalar() for _ in range(self.rows * self.columns)]
        if self.offset != len(self.data):
            raise ValueError(
                f"{self.path} has {len(self.data) - self.offset} trailing bytes"
            )
        return [
            values[row * self.columns : (row + 1) * self.columns]
            for row in range(self.rows)
        ]


Ast = Union[int, str, tuple[str, list["Ast"]]]


@dataclass
class FullFormParser:
    text: str
    offset: int = 0

    def _whitespace(self) -> None:
        while self.offset < len(self.text) and self.text[self.offset].isspace():
            self.offset += 1

    def expression(self) -> Ast:
        self._whitespace()
        start = self.offset
        if self.text[self.offset] == "-":
            self.offset += 1
        if self.offset < len(self.text) and self.text[self.offset].isdigit():
            while (
                self.offset < len(self.text)
                and self.text[self.offset].isdigit()
            ):
                self.offset += 1
            return int(self.text[start : self.offset])
        self.offset = start
        if self.text.startswith(r"\[", self.offset):
            close = self.text.find("]", self.offset + 2)
            if close < 0:
                raise ValueError(
                    f"unterminated named character at {self.text[self.offset:]}"
                )
            self.offset = close + 1
            while (
                self.offset < len(self.text)
                and self.text[self.offset].isdigit()
            ):
                self.offset += 1
        else:
            while (
                self.offset < len(self.text)
                and self.text[self.offset].isalnum()
            ):
                self.offset += 1
        name = self.text[start : self.offset]
        if not name:
            raise ValueError(
                f"unexpected FullForm token at {self.text[self.offset:]}"
            )
        self._whitespace()
        if self.offset >= len(self.text) or self.text[self.offset] != "[":
            return name
        self.offset += 1
        arguments: list[Ast] = []
        while True:
            arguments.append(self.expression())
            self._whitespace()
            if self.text[self.offset] == ",":
                self.offset += 1
                continue
            if self.text[self.offset] != "]":
                raise ValueError(
                    f"unterminated FullForm call at {self.text[self.offset:]}"
                )
            self.offset += 1
            return (name, arguments)

    def complete(self) -> Ast:
        result = self.expression()
        self._whitespace()
        if self.offset != len(self.text):
            raise ValueError(f"trailing FullForm input: {self.text[self.offset:]}")
        return result


def add(left: Polynomial, right: Polynomial) -> Polynomial:
    result: defaultdict[tuple[int, int], Fraction] = defaultdict(Fraction)
    result.update(left)
    for powers, coefficient in right.items():
        result[powers] += coefficient
    return {powers: coefficient for powers, coefficient in result.items() if coefficient}


def multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: defaultdict[tuple[int, int], Fraction] = defaultdict(Fraction)
    for (left_x, left_y), left_coefficient in left.items():
        for (right_x, right_y), right_coefficient in right.items():
            result[(left_x + right_x, left_y + right_y)] += (
                left_coefficient * right_coefficient
            )
    return {powers: coefficient for powers, coefficient in result.items() if coefficient}


def polynomial(ast: Ast) -> Polynomial:
    if isinstance(ast, int):
        return {(0, 0): Fraction(ast)}
    if isinstance(ast, str):
        if ast == "x":
            return {(1, 0): Fraction(1)}
        if ast == "y":
            return {(0, 1): Fraction(1)}
        raise ValueError(f"unsupported symbol in real-space B411 row: {ast}")
    head, arguments = ast
    if head == "Rational":
        if len(arguments) != 2 or not all(
            isinstance(value, int) for value in arguments
        ):
            raise ValueError("non-integer Rational in B411 coefficient")
        return {(0, 0): Fraction(arguments[0], arguments[1])}
    if head == "Plus":
        result: Polynomial = {}
        for argument in arguments:
            result = add(result, polynomial(argument))
        return result
    if head == "Times":
        result = {(0, 0): Fraction(1)}
        for argument in arguments:
            result = multiply(result, polynomial(argument))
        return result
    if head == "Power":
        if len(arguments) != 2 or not isinstance(arguments[1], int):
            raise ValueError("non-integer power in real-space B411 coefficient")
        base = polynomial(arguments[0])
        if len(base) != 1 or next(iter(base.values())) != 1:
            raise ValueError("non-monomial Power base in B411 coefficient")
        x_power, y_power = next(iter(base))
        exponent = arguments[1]
        return {(x_power * exponent, y_power * exponent): Fraction(1)}
    raise ValueError(f"unsupported FullForm head in B411 coefficient: {head}")


def frequency(value: Scalar) -> tuple[int, bool]:
    if isinstance(value, int):
        return value, False
    if not isinstance(value, str):
        raise ValueError("unexpected rational B411 frequency")
    if value == r"Times[Rational[-1, 2], \[Nu]]":
        return 0, True
    match = re.fullmatch(
        r"Plus\[(-?\d+), Times\[Rational\[-1, 2\], \\\[Nu\]\]\]",
        value,
    )
    if match is None:
        raise ValueError(f"unexpected B411 frequency expression: {value}")
    return int(match.group(1)), True


def affine_frequency(value: Scalar) -> tuple[int, int]:
    """Decode ``integer - Nu_i/2`` into (integer, zero-based FFTLog slot)."""

    if isinstance(value, int):
        return value, -1
    if not isinstance(value, str):
        raise ValueError("unexpected rational public-loop frequency")
    match = re.fullmatch(
        r"Times\[Rational\[-1, 2\], \\\[Nu\](\d*)\]",
        value,
    )
    if match is not None:
        suffix = match.group(1)
        return 0, (int(suffix) - 1 if suffix else 0)
    match = re.fullmatch(
        r"Plus\[(-?\d+), Times\[Rational\[-1, 2\], "
        r"\\\[Nu\](\d*)\]\]",
        value,
    )
    if match is None:
        raise ValueError(f"unexpected public-loop frequency expression: {value}")
    suffix = match.group(2)
    return int(match.group(1)), (int(suffix) - 1 if suffix else 0)


def coefficient_records(
    rows: list[list[Scalar]],
    source_rows: tuple[int, ...],
) -> list[tuple[int, int, int, int, Fraction]]:
    records: list[tuple[int, int, int, int, Fraction]] = []
    for operator_index, row_index in enumerate(source_rows):
        for frequency_index, value in enumerate(rows[row_index]):
            if value == 0:
                continue
            if isinstance(value, Fraction):
                terms = {(0, 0): value}
            elif isinstance(value, int):
                terms = {(0, 0): Fraction(value)}
            else:
                terms = polynomial(FullFormParser(value).complete())
            for (x_power, y_power), coefficient in sorted(terms.items()):
                records.append(
                    (
                        operator_index,
                        frequency_index,
                        x_power,
                        y_power,
                        coefficient,
                    )
                )
    return records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def single_compressed_expression(path: Path) -> str:
    data = path.read_bytes()
    if data[:4] != b"WDX1" or data[0x20:0x22] != b"\xe0\xff":
        raise ValueError(f"{path} is not the expected scalar WDX expression")
    length = int.from_bytes(data[0x22:0x2A], "little")
    start = 0x2A
    stop = start + length
    if stop != len(data):
        raise ValueError(f"{path} has an invalid compressed-expression length")
    return zlib.decompress(data[start:stop]).decode("utf-8")


def ast_symbols(ast: Ast) -> set[str]:
    if isinstance(ast, int):
        return set()
    if isinstance(ast, str):
        return {ast}
    result: set[str] = set()
    for argument in ast[1]:
        result.update(ast_symbols(argument))
    return result


def bias_powers(ast: Ast) -> Counter[str]:
    if isinstance(ast, int):
        return Counter()
    if isinstance(ast, str):
        return Counter({ast: 1}) if ast in UV_BIAS_INDEX else Counter()
    head, arguments = ast
    if (
        head == "Power"
        and len(arguments) == 2
        and isinstance(arguments[0], str)
        and arguments[0] in UV_BIAS_INDEX
        and isinstance(arguments[1], int)
    ):
        return Counter({arguments[0]: arguments[1]})
    if head == "Plus":
        if any(ast_symbols(argument) & UV_BIAS_INDEX.keys() for argument in arguments):
            raise ValueError("UV bias factor unexpectedly appears inside Plus")
        return Counter()
    result: Counter[str] = Counter()
    for argument in arguments:
        result.update(bias_powers(argument))
    return result


def uv_operator_index(ast: Ast) -> int:
    powers = bias_powers(ast)
    if powers["b1"] < 2:
        raise ValueError(f"B411 UV term lacks b1^2: {powers}")
    powers["b1"] -= 2
    powers += Counter()
    if sum(powers.values()) != 1:
        raise ValueError(f"B411 UV term is not b1^2 times one operator: {powers}")
    symbol = next(iter(powers))
    if powers[symbol] != 1:
        raise ValueError(f"invalid B411 UV operator power: {powers}")
    return UV_BIAS_INDEX[symbol]


UvInstruction = tuple[str, int, int, int]


def uv_postfix(ast: Ast) -> list[UvInstruction]:
    if isinstance(ast, int):
        return [("Constant", ast, 1, 0)]
    if isinstance(ast, str):
        if ast in UV_BIAS_INDEX:
            return [("Constant", 1, 1, 0)]
        if ast in {"k1", "k2", "k3"}:
            return [(ast.upper(), 0, 1, 0)]
        raise ValueError(f"unsupported symbol in real-space B411 UV term: {ast}")
    head, arguments = ast
    if head == "Rational":
        if len(arguments) != 2 or not all(
            isinstance(value, int) for value in arguments
        ):
            raise ValueError("non-integer Rational in B411 UV term")
        return [("Constant", arguments[0], arguments[1], 0)]
    if head in {"Plus", "Times"}:
        if not arguments:
            raise ValueError(f"empty {head} in B411 UV term")
        result = uv_postfix(arguments[0])
        opcode = "Add" if head == "Plus" else "Multiply"
        for argument in arguments[1:]:
            result.extend(uv_postfix(argument))
            result.append((opcode, 0, 1, 0))
        return result
    if head == "Power":
        if len(arguments) != 2 or not isinstance(arguments[1], int):
            raise ValueError("non-integer Power in B411 UV term")
        if (
            isinstance(arguments[0], str)
            and arguments[0] in UV_BIAS_INDEX
        ):
            return [("Constant", 1, 1, 0)]
        result = uv_postfix(arguments[0])
        result.append(("IntegerPower", 0, 1, arguments[1]))
        return result
    raise ValueError(f"unsupported FullForm head in B411 UV term: {head}")


def uv_programs(path: Path) -> tuple[list[tuple[int, int, int]], list[UvInstruction]]:
    expression = FullFormParser(single_compressed_expression(path)).complete()
    if not isinstance(expression, tuple) or expression[0] != "Plus":
        raise ValueError("B411 UV expression is not a top-level Plus")
    programs: list[tuple[int, int, int]] = []
    instructions: list[UvInstruction] = []
    seen: set[int] = set()
    for term in expression[1]:
        symbols = ast_symbols(term)
        if "f" in symbols:
            continue
        forbidden = symbols - set(UV_BIAS_INDEX) - {"k1", "k2", "k3"}
        if forbidden:
            raise ValueError(
                f"unexpected symbol in real-space B411 UV term: {sorted(forbidden)}"
            )
        operator_index = uv_operator_index(term)
        if operator_index in seen:
            raise ValueError(f"duplicate B411 UV operator {operator_index}")
        seen.add(operator_index)
        offset = len(instructions)
        program = uv_postfix(term)
        instructions.extend(program)
        programs.append((operator_index, offset, len(program)))
    expected = set(range(len(BIAS_ROWS))) - {2}
    if seen != expected:
        raise ValueError(
            f"unexpected real-space B411 UV operator set: {sorted(seen)}"
        )
    if any(
        abs(numerator) > 2**31 - 1
        or denominator <= 0
        or denominator > 2**31 - 1
        or not -128 <= exponent <= 127
        for _, numerator, denominator, exponent in instructions
    ):
        raise ValueError("B411 UV instruction exceeds generated integer storage")
    return sorted(programs), instructions


def generate(
    frequency_path: Path,
    coefficient_path: Path,
    uv_path: Path,
) -> str:
    if sha256(frequency_path) != EXPECTED_FREQUENCY_SHA256:
        raise ValueError("B411 frequency-table SHA256 mismatch")
    if sha256(coefficient_path) != EXPECTED_COEFFICIENT_SHA256:
        raise ValueError("B411 coefficient-table SHA256 mismatch")
    if sha256(uv_path) != EXPECTED_UV_SHA256:
        raise ValueError("B411 UV-table SHA256 mismatch")

    frequency_rows = WdxDenseArray(frequency_path).decode()
    coefficient_rows = WdxDenseArray(coefficient_path).decode()
    if (len(frequency_rows), len(frequency_rows[0])) != (112, 3):
        raise ValueError("unexpected B411 frequency-table shape")
    if (len(coefficient_rows), len(coefficient_rows[0])) != (70, 112):
        raise ValueError("unexpected B411 coefficient-table shape")

    frequencies: list[tuple[tuple[int, int, int], int]] = []
    for row in frequency_rows:
        decoded = [frequency(value) for value in row]
        slots = [index for index, (_, active) in enumerate(decoded) if active]
        if len(slots) != 1:
            raise ValueError("B411 frequency row does not contain one FFTLog slot")
        frequencies.append(
            (tuple(value for value, _ in decoded), slots[0])  # type: ignore[arg-type]
        )

    records: list[tuple[int, int, int, int, Fraction]] = []
    for operator_index, (_, row_index) in enumerate(BIAS_ROWS):
        for frequency_index, value in enumerate(coefficient_rows[row_index]):
            if value == 0:
                continue
            if isinstance(value, Fraction):
                terms = {(0, 0): value}
            elif isinstance(value, int):
                terms = {(0, 0): Fraction(value)}
            else:
                terms = polynomial(FullFormParser(value).complete())
            for (x_power, y_power), coefficient in sorted(terms.items()):
                records.append(
                    (
                        operator_index,
                        frequency_index,
                        x_power,
                        y_power,
                        coefficient,
                    )
                )
    if len(records) != 1637:
        raise ValueError(f"expected 1637 B411 coefficient records, got {len(records)}")
    programs, uv_instructions = uv_programs(uv_path)

    output = [
        "#ifndef MARISA_B_EFT_V2_GENERATED_B411_TABLE_H",
        "#define MARISA_B_EFT_V2_GENERATED_B411_TABLE_H",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace marisa_b_eft_v2 {",
        "namespace generated_b411 {",
        "",
        "// Generated from oliverphilcox/OneLoopBispectrum (GPL-3.0).",
        "// Upstream table SHA256:",
        f'inline constexpr char kFrequencySha256[]="{EXPECTED_FREQUENCY_SHA256}";',
        f'inline constexpr char kCoefficientSha256[]="{EXPECTED_COEFFICIENT_SHA256}";',
        f'inline constexpr char kUvSha256[]="{EXPECTED_UV_SHA256}";',
        "",
        "struct FrequencyTerm {",
        "    std::array<std::int8_t,3> shift;",
        "    std::uint8_t fftlog_slot;",
        "};",
        "",
        "struct CoefficientTerm {",
        "    std::uint8_t operator_index;",
        "    std::uint8_t frequency_index;",
        "    std::int8_t x_power;",
        "    std::int8_t y_power;",
        "    std::int32_t numerator;",
        "    std::int32_t denominator;",
        "};",
        "",
        "inline constexpr std::array<FrequencyTerm,112> kFrequencies={{",
    ]
    for shifts, slot in frequencies:
        output.append(
            "    FrequencyTerm{{{%d,%d,%d}},%d}," % (*shifts, slot)
        )
    output.extend(
        [
            "}};",
            "",
            "inline constexpr std::array<CoefficientTerm,1637> kCoefficients={{",
        ]
    )
    for operator_index, frequency_index, x_power, y_power, value in records:
        output.append(
            "    CoefficientTerm{{{},{},{},{},{},{}}},"
            .format(
                operator_index,
                frequency_index,
                x_power,
                y_power,
                value.numerator,
                value.denominator,
            )
        )
    output.extend(
        [
            "}};",
            "",
            "enum class UvOpcode : std::uint8_t {",
            "    Constant, K1, K2, K3, Add, Multiply, IntegerPower,",
            "};",
            "",
            "struct UvInstruction {",
            "    UvOpcode opcode;",
            "    std::int32_t numerator;",
            "    std::int32_t denominator;",
            "    std::int8_t exponent;",
            "};",
            "",
            "struct UvProgram {",
            "    std::uint8_t operator_index;",
            "    std::uint16_t offset;",
            "    std::uint16_t count;",
            "};",
            "",
            f"inline constexpr std::array<UvInstruction,{len(uv_instructions)}> "
            "kUvInstructions={{",
        ]
    )
    for opcode, numerator, denominator, exponent in uv_instructions:
        output.append(
            "    UvInstruction{UvOpcode::%s,%d,%d,%d},"
            % (opcode, numerator, denominator, exponent)
        )
    output.extend(
        [
            "}};",
            "",
            f"inline constexpr std::array<UvProgram,{len(programs)}> kUvPrograms={{{{",
        ]
    )
    for operator_index, offset, count in programs:
        output.append(
            f"    UvProgram{{{operator_index},{offset},{count}}},"
        )
    output.extend(
        [
            "}};",
            "",
            "}  // namespace generated_b411",
            "}  // namespace marisa_b_eft_v2",
            "",
            "#endif",
            "",
        ]
    )
    return "\n".join(output)


def generate_b222_b321i(
    b222_frequency_path: Path,
    b222_coefficient_path: Path,
    b321i_frequency_path: Path,
    b321i_coefficient_path: Path,
) -> str:
    expected_hashes = (
        (b222_frequency_path, EXPECTED_B222_FREQUENCY_SHA256),
        (b222_coefficient_path, EXPECTED_B222_COEFFICIENT_SHA256),
        (b321i_frequency_path, EXPECTED_B321I_FREQUENCY_SHA256),
        (b321i_coefficient_path, EXPECTED_B321I_COEFFICIENT_SHA256),
    )
    for path, expected in expected_hashes:
        if sha256(path) != expected:
            raise ValueError(f"public loop table SHA256 mismatch: {path}")

    b222_frequencies_raw = WdxDenseArray(b222_frequency_path).decode()
    b222_coefficients_raw = WdxDenseArray(b222_coefficient_path).decode()
    b321i_frequencies_raw = WdxDenseArray(b321i_frequency_path).decode()
    b321i_coefficients_raw = WdxDenseArray(b321i_coefficient_path).decode()
    if (len(b222_frequencies_raw), len(b222_frequencies_raw[0])) != (80, 3):
        raise ValueError("unexpected B222 frequency-table shape")
    if (len(b222_coefficients_raw), len(b222_coefficients_raw[0])) != (50, 80):
        raise ValueError("unexpected B222 coefficient-table shape")
    if (len(b321i_frequencies_raw), len(b321i_frequencies_raw[0])) != (155, 3):
        raise ValueError("unexpected B321I frequency-table shape")
    if (len(b321i_coefficients_raw), len(b321i_coefficients_raw[0])) != (86, 155):
        raise ValueError("unexpected B321I coefficient-table shape")

    b222_frequencies = [
        tuple(affine_frequency(value) for value in row)
        for row in b222_frequencies_raw
    ]
    b321i_frequencies = [
        tuple(affine_frequency(value) for value in row)
        for row in b321i_frequencies_raw
    ]
    if any(
        tuple(slot for _, slot in row) != (0, 1, 2)
        for row in b222_frequencies
    ):
        raise ValueError("B222 frequency table does not use Nu1,Nu2,Nu3 in order")
    if any(
        slot not in {-1, 0, 1}
        for row in b321i_frequencies
        for _, slot in row
    ):
        raise ValueError("B321I frequency table uses an unexpected FFTLog slot")

    b222_records = coefficient_records(
        b222_coefficients_raw, B222_REALSPACE_ROWS
    )
    b321i_records = coefficient_records(
        b321i_coefficients_raw, B321I_REALSPACE_ROWS
    )
    if len(b222_records) != 1026:
        raise ValueError(
            f"expected 1026 real-space B222 records, got {len(b222_records)}"
        )
    if len(b321i_records) != 2523:
        raise ValueError(
            f"expected 2523 real-space B321I records, got {len(b321i_records)}"
        )
    all_records = b222_records + b321i_records
    if any(
        operator > 255
        or frequency_index > 255
        or not -128 <= x_power <= 127
        or not -128 <= y_power <= 127
        or abs(value.numerator) > 2**31 - 1
        or value.denominator > 2**31 - 1
        for operator, frequency_index, x_power, y_power, value in all_records
    ):
        raise ValueError("generated public-loop coefficient exceeds C++ storage")

    output = [
        "#ifndef MARISA_B_EFT_V2_GENERATED_B222_B321I_TABLES_H",
        "#define MARISA_B_EFT_V2_GENERATED_B222_B321I_TABLES_H",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace marisa_b_eft_v2 {",
        "namespace generated_public_loops {",
        "",
        "// Generated from oliverphilcox/OneLoopBispectrum (GPL-3.0).",
        f'inline constexpr char kB222FrequencySha256[]="{EXPECTED_B222_FREQUENCY_SHA256}";',
        f'inline constexpr char kB222CoefficientSha256[]="{EXPECTED_B222_COEFFICIENT_SHA256}";',
        f'inline constexpr char kB321IFrequencySha256[]="{EXPECTED_B321I_FREQUENCY_SHA256}";',
        f'inline constexpr char kB321ICoefficientSha256[]="{EXPECTED_B321I_COEFFICIENT_SHA256}";',
        "",
        "struct AffineFrequency {",
        "    std::array<std::int8_t,3> shift;",
        "    std::array<std::int8_t,3> fftlog_slot;",
        "};",
        "",
        "struct CoefficientTerm {",
        "    std::uint8_t operator_index;",
        "    std::uint8_t frequency_index;",
        "    std::int8_t x_power;",
        "    std::int8_t y_power;",
        "    std::int32_t numerator;",
        "    std::int32_t denominator;",
        "};",
        "",
        "inline constexpr std::array<std::uint8_t,10> kB222SourceRows={{",
        "    " + ",".join(str(value) for value in B222_REALSPACE_ROWS),
        "}};",
        "inline constexpr std::array<std::uint8_t,18> kB321ISourceRows={{",
        "    " + ",".join(str(value) for value in B321I_REALSPACE_ROWS),
        "}};",
        "",
        "inline constexpr std::array<AffineFrequency,80> kB222Frequencies={{",
    ]

    def append_frequencies(
        values: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]
    ) -> None:
        for row in values:
            shifts = ",".join(str(value[0]) for value in row)
            slots = ",".join(str(value[1]) for value in row)
            output.append(f"    AffineFrequency{{{{{shifts}}},{{{slots}}}}},")

    def append_records(
        values: list[tuple[int, int, int, int, Fraction]]
    ) -> None:
        for operator, frequency_index, x_power, y_power, value in values:
            output.append(
                "    CoefficientTerm{%d,%d,%d,%d,%d,%d},"
                % (
                    operator,
                    frequency_index,
                    x_power,
                    y_power,
                    value.numerator,
                    value.denominator,
                )
            )

    append_frequencies(b222_frequencies)  # type: ignore[arg-type]
    output.extend(
        [
            "}};",
            "",
            "inline constexpr std::array<CoefficientTerm,1026> "
            "kB222Coefficients={{",
        ]
    )
    append_records(b222_records)
    output.extend(
        [
            "}};",
            "",
            "inline constexpr std::array<AffineFrequency,155> "
            "kB321IFrequencies={{",
        ]
    )
    append_frequencies(b321i_frequencies)  # type: ignore[arg-type]
    output.extend(
        [
            "}};",
            "",
            "inline constexpr std::array<CoefficientTerm,2523> "
            "kB321ICoefficients={{",
        ]
    )
    append_records(b321i_records)
    output.extend(
        [
            "}};",
            "",
            "}  // namespace generated_public_loops",
            "}  // namespace marisa_b_eft_v2",
            "",
            "#endif",
            "",
        ]
    )
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", type=Path)
    parser.add_argument("--coefficients", type=Path)
    parser.add_argument("--uv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--b222-frequency", type=Path)
    parser.add_argument("--b222-coefficients", type=Path)
    parser.add_argument("--b321i-frequency", type=Path)
    parser.add_argument("--b321i-coefficients", type=Path)
    parser.add_argument("--loop-output", type=Path)
    arguments = parser.parse_args()
    b411_arguments = (
        arguments.frequency,
        arguments.coefficients,
        arguments.uv,
        arguments.output,
    )
    loop_arguments = (
        arguments.b222_frequency,
        arguments.b222_coefficients,
        arguments.b321i_frequency,
        arguments.b321i_coefficients,
        arguments.loop_output,
    )
    if any(value is not None for value in b411_arguments):
        if any(value is None for value in b411_arguments):
            parser.error(
                "--frequency, --coefficients, --uv and --output are one group"
            )
        rendered = generate(
            arguments.frequency,
            arguments.coefficients,
            arguments.uv,
        )
        previous = (
            arguments.output.read_text(encoding="utf-8")
            if arguments.output.exists()
            else None
        )
        if rendered != previous:
            arguments.output.write_text(rendered, encoding="utf-8")
    if any(value is not None for value in loop_arguments):
        if any(value is None for value in loop_arguments):
            parser.error(
                "--b222-frequency, --b222-coefficients, --b321i-frequency, "
                "--b321i-coefficients and --loop-output are one group"
            )
        rendered = generate_b222_b321i(
            arguments.b222_frequency,
            arguments.b222_coefficients,
            arguments.b321i_frequency,
            arguments.b321i_coefficients,
        )
        previous = (
            arguments.loop_output.read_text(encoding="utf-8")
            if arguments.loop_output.exists()
            else None
        )
        if rendered != previous:
            arguments.loop_output.write_text(rendered, encoding="utf-8")
    if not any(value is not None for value in b411_arguments + loop_arguments):
        parser.error("no output table group was requested")


if __name__ == "__main__":
    main()
