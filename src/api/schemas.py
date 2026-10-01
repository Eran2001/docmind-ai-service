from typing import Annotated

from pydantic import StringConstraints

type Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
