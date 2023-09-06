import copy


class Peak:
    def __init__(self, chrom: str, start: int, end: int):
        assert (start >= 0) and end > start
        self._chrom = chrom
        self._start = start
        self._end = end

    @property
    def chrom(self):
        return self._chrom

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def width(self):
        return self.end - self.start

    @property
    def center(self):
        return self.start + self.width // 2

    @classmethod
    def from_str(cls, s):
        chrom, start_end = s.split(":")
        start, end = start_end.split("-")
        return cls(chrom=chrom, start=int(start), end=int(end))

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return self.width

    def __str__(self):
        return f"{self.chrom}:{self.start}-{self.end}"

    def __repr__(self):
        return f"Peak({self.chrom}, {self.start}, {self.end})"

    def __eq__(self, other):
        return (
            (self.chrom == other.chrom)
            and (self.start == other.start)
            and (self.end == other.end)
        )
