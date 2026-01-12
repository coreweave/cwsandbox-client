# CLI Reference

The Aviato CLI provides command-line access to manage sandboxes.

## Installation

The CLI is included when you install the aviato package:

```bash
pip install aviato
```

## Authentication

The CLI uses the same authentication as the SDK. Set one of:

```bash
# Option 1: Aviato API Key (recommended)
export AVIATO_API_KEY="your-api-key"

# Option 2: W&B credentials
export WANDB_API_KEY="your-wandb-key"
export WANDB_ENTITY_NAME="your-entity"
```

## Commands

### aviato sandbox list

List sandboxes with optional filters.

**Usage:**

```bash
aviato sandbox list [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-s, --status STATUS` | Filter by status (running, completed, failed, terminated, etc.) |
| `-t, --tag TAG` | Filter by tag (can be repeated) |
| `-o, --output FORMAT` | Output format: `table` (default), `json`, or `quiet` |
| `-v, --verbose` | Show additional columns in table output |

---

#### Table Output (Default)

```bash
aviato sandbox list
```

```
ID                                    STATUS     AGE
a1b2c3d4-5678-90ab-cdef-1234567890ab  running    5m
e5f6a7b8-9012-34cd-ef56-7890abcdef12  running    2h
12345678-abcd-ef01-2345-6789abcdef01  completed  1d
```

---

#### Table Output with Verbose

```bash
aviato sandbox list -v
```

```
ID                                    STATUS     AGE   IMAGE
a1b2c3d4-5678-90ab-cdef-1234567890ab  running    5m    python:3.11
e5f6a7b8-9012-34cd-ef56-7890abcdef12  running    2h    pytorch/pytorch:latest
12345678-abcd-ef01-2345-6789abcdef01  completed  1d    python:3.11
```

---

#### JSON Output

```bash
aviato sandbox list -o json
```

```json
[
  {
    "id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
    "status": "running",
    "started_at": "2025-01-12T10:30:00+00:00"
  },
  {
    "id": "e5f6a7b8-9012-34cd-ef56-7890abcdef12",
    "status": "running",
    "started_at": "2025-01-12T08:15:00+00:00"
  },
  {
    "id": "12345678-abcd-ef01-2345-6789abcdef01",
    "status": "completed",
    "started_at": "2025-01-11T10:00:00+00:00"
  }
]
```

---

#### Quiet Output

Returns only sandbox IDs, one per line. Useful for scripting:

```bash
aviato sandbox list -o quiet
```

```
a1b2c3d4-5678-90ab-cdef-1234567890ab
e5f6a7b8-9012-34cd-ef56-7890abcdef12
12345678-abcd-ef01-2345-6789abcdef01
```

---

#### Filter by Status

```bash
aviato sandbox list --status running
```

```
ID                                    STATUS   AGE
a1b2c3d4-5678-90ab-cdef-1234567890ab  running  5m
e5f6a7b8-9012-34cd-ef56-7890abcdef12  running  2h
```

Available statuses: `running`, `pending`, `creating`, `completed`, `failed`, `terminated`, `paused`, `unspecified`

---

#### Filter by Tag

```bash
aviato sandbox list --tag my-project
```

```
ID                                    STATUS   AGE
a1b2c3d4-5678-90ab-cdef-1234567890ab  running  5m
```

Filter by multiple tags (sandboxes must have ALL specified tags):

```bash
aviato sandbox list --tag my-project --tag batch-job
```

---

#### Empty Results

When no sandboxes match the filters:

```bash
aviato sandbox list --tag nonexistent-tag
```

```
No sandboxes found.
```
