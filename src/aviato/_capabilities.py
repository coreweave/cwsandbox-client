"""Infrastructure capabilities discovery API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from aviato._auth import resolve_auth
from aviato._defaults import DEFAULT_BASE_URL, DEFAULT_REQUEST_TIMEOUT_SECONDS

import httpx
from coreweave.aviato.v1beta1 import atc_connect, atc_pb2


@dataclass(frozen=True)
class RunwayInfo:
    """Information about a specific runway on a specific tower."""
    
    tower_id: str
    runway_id: str
    gpu_types: list[str]
    max_gpu_count: int
    max_cpu_millicores: int
    max_memory_bytes: int
    supports_service_exposure: bool
    
    @property
    def max_cpu_cores(self) -> float:
        """Maximum CPU in cores."""
        return self.max_cpu_millicores / 1000.0
    
    @property
    def max_memory_gb(self) -> float:
        """Maximum memory in GB."""
        return self.max_memory_bytes / (1024.0 ** 3)
    
    @property
    def supports_gpu(self) -> bool:
        """Check if runway supports GPUs."""
        return len(self.gpu_types) > 0
    
    def has_gpu_type(self, gpu_type: str) -> bool:
        """Check if a specific GPU type is available on this runway.
        
        Args:
            gpu_type: GPU type to check (e.g., "A100", "A40")
            
        Returns:
            True if the GPU type is available
            
        Example:
            if runway.has_gpu_type("A100"):
                print("A100 available on this runway")
        """
        return gpu_type in self.gpu_types
    
    def __str__(self) -> str:
        gpu_info = f"{', '.join(self.gpu_types)} (max {self.max_gpu_count})" if self.supports_gpu else "No GPU"
        return (
            f"Runway '{self.runway_id}' on Tower '{self.tower_id}': "
            f"GPU: {gpu_info}, "
            f"CPU: {self.max_cpu_cores:.1f} cores, "
            f"Memory: {self.max_memory_gb:.1f} GB"
        )


@dataclass(frozen=True)
class Capabilities:
    """Infrastructure capabilities."""
    
    runways: list[RunwayInfo]
    queried_at: datetime
    
    def get_runway(self, tower_id: str, runway_id: str) -> Optional[RunwayInfo]:
        """Get information about a specific runway on a specific tower.
        
        Args:
            tower_id: Tower identifier
            runway_id: Runway identifier (e.g., "gpu-a100", "cpu")
            
        Returns:
            RunwayInfo if found, None otherwise
            
        Example:
            runway = caps.get_runway("tower-us-east", "gpu-a100")
            if runway:
                print(f"Max GPUs: {runway.max_gpu_count}")
        """
        for runway in self.runways:
            if runway.tower_id == tower_id and runway.runway_id == runway_id:
                return runway
        return None
    
    def find_runways_by_name(self, runway_id: str) -> list[RunwayInfo]:
        """Find all runways with a given name across all towers.
        
        Args:
            runway_id: Runway identifier (e.g., "gpu-a100")
            
        Returns:
            List of matching runways from different towers
            
        Example:
            # Find all "gpu-a100" runways across all towers
            a100_runways = caps.find_runways_by_name("gpu-a100")
            for runway in a100_runways:
                print(f"Tower {runway.tower_id}: {runway.max_gpu_count} GPUs")
        """
        return [r for r in self.runways if r.runway_id == runway_id]
    
    def find_runways_with_gpu(self, gpu_type: Optional[str] = None) -> list[RunwayInfo]:
        """Find all runways that support GPU, optionally filtered by type.
        
        Args:
            gpu_type: Optional GPU type to filter by (e.g., "A100")
            
        Returns:
            List of matching runways
            
        Example:
            # Find all GPU runways
            gpu_runways = caps.find_runways_with_gpu()
            
            # Find A100 runways specifically
            a100_runways = caps.find_runways_with_gpu("A100")
        """
        runways = [r for r in self.runways if r.supports_gpu]
        
        if gpu_type:
            runways = [r for r in runways if r.has_gpu_type(gpu_type)]
        
        return runways
    
    @property
    def available_gpu_types(self) -> set[str]:
        """Get all GPU types available across all runways."""
        return set(r.gpu_types for r in self.runways)
    
    @classmethod
    async def get(
        cls,
        *,
        runway_ids: Optional[list[str]] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Capabilities:
        """Query infrastructure capabilities from ATC.
        
        Args:
            runway_ids: Optional list of specific runway IDs to query
            base_url: Aviato API URL (default: AVIATO_BASE_URL env or localhost)
            timeout_seconds: Request timeout
            
        Returns:
            Capabilities object with infrastructure information
            
        Example:
            # Get all capabilities
            caps = await Capabilities.get()
            
            # Get specific runways
            caps = await Capabilities.get(runway_ids=["gpu-a100", "cpu"])
            
            # Print summary
            for runway in caps.runways:
                print(runway)
        """
        auth = resolve_auth()
        effective_base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        timeout = timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS
        
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=auth.headers,
        ) as http_client:
            client = atc_connect.ATCServiceClient(
                address=effective_base_url,
                session=http_client,
                proto_json=True,
            )
            
            # TODO: This API is not yet implemented in the backend.
            # Proposed mock response: 
            #
            # {
            #   "runways": [
            #     {
            #       "tower_id": "tower-us-east-1",
            #       "runway_id": "gpu-a100",
            #       "gpu_types": ["A100"],
            #       "max_gpu_count": 8,
            #       "max_cpu_millicores": 128000,
            #       "max_memory_bytes": 549755813888,
            #       "supports_service_exposure": true
            #     },
            #     {
            #       "tower_id": "tower-us-east-1",
            #       "runway_id": "cpu",
            #       "gpu_types": [],
            #       "max_gpu_count": 0,
            #       "max_cpu_millicores": 64000,
            #       "max_memory_bytes": 274877906944,
            #       "supports_service_exposure": false
            #     },
            #     {
            #       "tower_id": "tower-eu-west-1",
            #       "runway_id": "gpu-a100",
            #       "gpu_types": ["A100"],
            #       "max_gpu_count": 4,
            #       "max_cpu_millicores": 64000,
            #       "max_memory_bytes": 274877906944,
            #       "supports_service_exposure": true
            #     }
            #   ],
            #   "queried_at": "2024-06-10T12:34:56Z"
            # }
            
            
            request = atc_pb2.GetCapabilitiesRequest(
                runway_ids=runway_ids or [],
            )
            
            response = await client.get_capabilities(request)
            
            runways = [
                RunwayInfo(
                    tower_id=r.tower_id,
                    runway_id=r.runway_id,
                    gpu_types=list(r.gpu_types),
                    max_gpu_count=r.max_gpu_count,
                    max_cpu_millicores=r.max_cpu_millicores,
                    max_memory_bytes=r.max_memory_bytes,
                    supports_service_exposure=r.supports_service_exposure,
                )
                for r in response.runways
            ]
            
            return cls(
                runways=runways,
                queried_at=response.queried_at.ToDatetime(),
            )

