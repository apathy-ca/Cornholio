#!/usr/bin/env python3
"""
Score state manager: applies cornhole rules, cancel-out logic, running totals.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RoundResult:
    round_num: int
    timestamp_sec: float
    red_cornholes: int
    red_on_board: int
    blue_cornholes: int
    blue_on_board: int
    red_raw: int = 0
    blue_raw: int = 0
    red_net: int = 0
    blue_net: int = 0
    red_total: int = 0
    blue_total: int = 0
    flagged: bool = False
    flag_reason: str = ""

    def __post_init__(self):
        self.red_raw = 3 * self.red_cornholes + self.red_on_board
        self.blue_raw = 3 * self.blue_cornholes + self.blue_on_board
        if self.red_raw > self.blue_raw:
            self.red_net = self.red_raw - self.blue_raw
            self.blue_net = 0
        elif self.blue_raw > self.red_raw:
            self.blue_net = self.blue_raw - self.red_raw
            self.red_net = 0
        else:
            self.red_net = 0
            self.blue_net = 0


class ScoreState:
    WINNING_SCORE = 21

    def __init__(self):
        self.red_total = 0
        self.blue_total = 0
        self.rounds: list[RoundResult] = []
        self.game_over = False
        self.winner: Optional[str] = None

    @property
    def round_num(self) -> int:
        return len(self.rounds) + 1

    def record_round(
        self,
        timestamp_sec: float,
        red_cornholes: int,
        red_on_board: int,
        blue_cornholes: int,
        blue_on_board: int,
        flagged: bool = False,
        flag_reason: str = "",
    ) -> RoundResult:
        result = RoundResult(
            round_num=self.round_num,
            timestamp_sec=timestamp_sec,
            red_cornholes=red_cornholes,
            red_on_board=red_on_board,
            blue_cornholes=blue_cornholes,
            blue_on_board=blue_on_board,
            flagged=flagged,
            flag_reason=flag_reason,
        )
        self.red_total += result.red_net
        self.blue_total += result.blue_net
        result.red_total = self.red_total
        result.blue_total = self.blue_total
        self.rounds.append(result)

        if self.red_total >= self.WINNING_SCORE:
            self.game_over = True
            self.winner = "red"
        elif self.blue_total >= self.WINNING_SCORE:
            self.game_over = True
            self.winner = "blue"

        return result

    def announce_text(self, result: RoundResult) -> str:
        parts = []
        if result.red_net > 0:
            parts.append(f"Red scores {result.red_net}.")
        elif result.blue_net > 0:
            parts.append(f"Blue scores {result.blue_net}.")
        else:
            parts.append("No points this round.")

        parts.append(f"Score: Red {result.red_total}, Blue {result.blue_total}.")

        if self.game_over:
            parts.append(f"{self.winner.title()} wins the game!")
        if result.flagged:
            parts.append(f"Note: {result.flag_reason}")

        return " ".join(parts)

    def save_log(self, path: str):
        data = {
            "red_total": self.red_total,
            "blue_total": self.blue_total,
            "winner": self.winner,
            "rounds": [asdict(r) for r in self.rounds],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Score log saved to {path}")

    def summary(self) -> str:
        lines = [f"{'Round':>6}  {'Time':>8}  {'R-CH':>4}  {'R-OB':>4}  "
                 f"{'B-CH':>4}  {'B-OB':>4}  {'R-Net':>6}  {'B-Net':>6}  "
                 f"{'R-Tot':>6}  {'B-Tot':>6}  Flag"]
        lines.append("-" * 80)
        for r in self.rounds:
            mins = int(r.timestamp_sec // 60)
            secs = r.timestamp_sec % 60
            flag = "!" if r.flagged else ""
            lines.append(
                f"{r.round_num:>6}  {mins:02d}:{secs:04.1f}  "
                f"{r.red_cornholes:>4}  {r.red_on_board:>4}  "
                f"{r.blue_cornholes:>4}  {r.blue_on_board:>4}  "
                f"{r.red_net:>6}  {r.blue_net:>6}  "
                f"{r.red_total:>6}  {r.blue_total:>6}  {flag}"
            )
        lines.append("-" * 80)
        lines.append(f"Final: Red {self.red_total}  Blue {self.blue_total}"
                     + (f"  Winner: {self.winner}" if self.winner else ""))
        return "\n".join(lines)
