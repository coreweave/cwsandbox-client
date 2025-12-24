"""Example: Discover capabilities and create a sandbox on a specific runway.

This example demonstrates:
- Querying available towers and runways
- Finding runways with specific GPU types
- Filtering by runway name
- Pre-flight validation before creating a sandbox
"""

import asyncio

from aviato import Capabilities, Sandbox


async def main() -> None:
    # Query all available capabilities
    caps = await Capabilities.get()
    
    print(f"Queried at: {caps.queried_at}")
    print(f"Found {len(caps.runways)} runway(s) across all towers\n")
    
    # Display all runways
    print("\nALL AVAILABLE RUNWAYS:\n")
    for runway in caps.runways:
        print(runway)
    
    # Find all GPU runways
    print("\nGPU RUNWAYS:\n")
    gpu_runways = caps.find_runways_with_gpu()
    for runway in gpu_runways:
        print(runway)
    
    # Find A100 runways specifically
    print("\nA100 RUNWAYS:\n")
    a100_runways = caps.find_runways_with_gpu("A100")
    for runway in a100_runways:
        print(runway)
    
    # Show all unique GPU types
    print("\nALL GPU TYPES:\n")
    gpu_types = caps.available_gpu_types
    print(gpu_types)
    
    # Find all runways with a specific name across towers
    print("\nRUNWAYS BY NAME:\n")
    gpu_a100_runways = caps.find_runways_by_name("gpu-a100")
    for runway in gpu_a100_runways:
        print(runway)
    
    # Pre-flight check: verify a specific runway exists before creating sandbox
    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECK")
    print("=" * 70)
    
    # Example: Check if we can create a sandbox on a specific tower and runway
    desired_gpu_count = 4
    if gpu_a100_runways and gpu_a100_runways[0].max_gpu_count >= desired_gpu_count:
        async with Sandbox(
            command="python",
            args=["-c", "import torch; print(f'GPUs: {torch.cuda.device_count()}')"],
            runway_ids=["gpu-a100"],
            resources={"gpu": {"gpu_count": desired_gpu_count, "gpu_type": "A100"}},
        ) as sandbox:
            result = await sandbox.exec(command=["python", "-c", "print(2+2)"])
            print(f"Sandbox output: {result.stdout}")


if __name__ == "__main__":
    asyncio.run(main())
