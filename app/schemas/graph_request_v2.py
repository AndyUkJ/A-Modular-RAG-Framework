from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Sentence(BaseModel):
    id: str
    text: str


class ProvenanceItem(BaseModel):
    id: str
    source: str
    extra: Dict[str, Any] = Field(default_factory=dict)


class Inputs(BaseModel):
    sentences: List[Sentence] = Field(default_factory=list)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class Provenance(BaseModel):
    items: List[ProvenanceItem] = Field(default_factory=list)


class AssembleGraphRequestV2(BaseModel):
    graph_id: str
    inputs: Inputs
    provenance: Provenance = Field(default_factory=Provenance)
