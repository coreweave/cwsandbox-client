# 7. Remote Function Execution

Run Python functions in sandboxes without writing command strings:

```python
from cwsandbox import Sandbox, SandboxDefaults

with Sandbox.session(SandboxDefaults()) as session:
    @session.function()
    def add(x: int, y: int) -> int:
        return x + y

    result = add.remote(2, 3).result()  # 5
```

Parallel execution with `.map()`:

```python
@session.function()
def square(x: int) -> int:
    return x * x

refs = square.map([(1,), (2,), (3,)])
results = [r.result() for r in refs]  # [1, 4, 9]
```

For serialization modes, `.local()` testing, and more, see the [Remote Functions Guide](../guides/remote-functions.md).
