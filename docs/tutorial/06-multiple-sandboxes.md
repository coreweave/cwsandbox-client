# 6. Managing Multiple Sandboxes

Sessions manage multiple sandboxes with shared defaults and automatic cleanup:

```python
from cwsandbox import Sandbox, SandboxDefaults

defaults = SandboxDefaults(container_image="python:3.11", tags=("my-app",))

with Sandbox.session(defaults) as session:
    sb1 = session.sandbox()
    sb2 = session.sandbox()

    p1 = sb1.exec(["echo", "one"])
    p2 = sb2.exec(["echo", "two"])

    print(p1.result().stdout, p2.result().stdout)
# All sandboxes cleaned up
```

For sandbox pools, adoption patterns, and lifecycle management, see the [Sessions Guide](../guides/sessions.md).

