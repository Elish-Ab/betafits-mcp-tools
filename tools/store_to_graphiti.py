from graphiti_client import graphiti, initialize_indicies_and_constraints
from langchain_core.tools import StructuredTool
from graphiti_core.nodes import EpisodeType
import asyncio
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import logging
from typing import Optional
import sys

logger = logging.getLogger(__name__)


# Content Provided to the chain
class Content(BaseModel):
    core: str = Field(description="The knowledge graph root.")
    domain: str = Field(description="Which domain of the root.")
    project: str = Field(description="Tell about the Project. Which project.")
    component: str = Field(description="Which component of the project.")
    feature: str = Field(description="Which feature of the project we talking.")
    description: str = Field(description="description of the content.")


# ===================
# Custom entity types
# ===================
class Core(BaseModel):
    """Root level of the Knowledge graph - the organization"""

    description: Optional[str] = Field(
        description="Description of the core organization"
    )
    founded_year: Optional[int] = Field(
        None, description="Year the organization was founded"
    )
    website: Optional[str] = Field(None, description="Organization website")


class Domain(BaseModel):
    """Top-level division or domain within the organization"""

    description: Optional[str] = Field(description="Description of the domain")
    tags: Optional[str] = Field(
        None, description="Key tags for the description of the domain"
    )
    purpose: Optional[str] = Field(None, description="Purpose of the domain")


class Project(BaseModel):
    """A project within a domain"""

    description: Optional[str] = Field(None, description="Project Description")
    start_date: Optional[datetime] = Field(None, description="Project start date")
    end_date: Optional[datetime] = Field(None, description="Project end date")


class Component(BaseModel):
    """A component or service within a project"""

    description: Optional[str] = Field(None, description="Component Description")


class Feature(BaseModel):
    """Some Functionality in the Project."""

    description: Optional[str] = Field(None, description="Component description")


# ===============
# Custom Edges
# ===============


class HasDomain(BaseModel):
    """Hierarchical relationship from Core to Domain."""

    description: Optional[str] = Field(None, description="Why this domain exists")


class HasProject(BaseModel):
    """Hierarchical relationship from Domain to Project."""

    justification: Optional[str] = Field(
        None, description="Why this project was created"
    )


class HasComponent(BaseModel):
    """Hierarchical relationship from Project to Component."""

    responsibility: Optional[str] = Field(
        None, description="Team responsible for this component"
    )


class HasFeature(BaseModel):
    """Hierarchical relationship from Component to Feature."""

    release_date: Optional[datetime] = Field(None, description="Feature release date")
    version: Optional[str] = Field(None, description="Version when feature was added")
    impact: Optional[str] = Field(None, description="Expected impact of feature")


class DependsOn(BaseModel):
    """Dependency relationship between features or components."""

    dependency_type: Optional[str] = Field(
        None, description="Type of dependency (hard, soft, optional)"
    )
    reason: Optional[str] = Field(None, description="Reason for dependency")
    identified_date: Optional[datetime] = Field(
        None, description="When dependency was identified"
    )
    strength: Optional[float] = Field(None, description="Dependency strength (0-1)")


class IntegratesWith(BaseModel):
    """Integration relationship between components."""

    integration_type: Optional[str] = Field(
        None, description="Type of integration (API, Database, Message Queue)"
    )
    protocol: Optional[str] = Field(
        None, description="Integration protocol (REST, gRPC, AMQP)"
    )
    data_flow: Optional[str] = Field(None, description="Direction of data flow")
    integration_date: Optional[datetime] = Field(
        None, description="When integration was established"
    )


class MentionedIn(BaseModel):
    """Link between PCF records and features/components."""

    context: Optional[str] = Field(
        None, description="Context in which it was mentioned"
    )
    relevance: Optional[str] = Field(None, description="Relevance to the meeting")
    action_required: Optional[bool] = Field(
        None, description="Whether action is required"
    )
    timestamp: Optional[datetime] = Field(None, description="When it was mentioned")


# Define entity types dictionary
entity_types = {
    "Core": Core,
    "Domain": Domain,
    "Project": Project,
    "Component": Component,
    "Feature": Feature,
}

# Define edge types dictionary
edge_types = {
    "HasDomain": HasDomain,
    "HasProject": HasProject,
    "HasComponent": HasComponent,
    "HasFeature": HasFeature,
    "DependsOn": DependsOn,
    "IntegratesWith": IntegratesWith,
    "MentionedIn": MentionedIn,
}

# Define edge type mapping (which edges can exist between which entities)
edge_type_map = {
    # Hierarchical relationships
    ("Core", "Domain"): ["HasDomain"],
    ("Domain", "Project"): ["HasProject"],
    ("Project", "Component"): ["HasComponent"],
    ("Component", "Component"): ["HasComponent"],  # Sub-components
    ("Component", "Feature"): ["HasFeature"],
    # Cross-hierarchical relationships
    ("Feature", "Feature"): ["DependsOn"],
    ("Component", "Feature"): ["DependsOn"],
    ("Feature", "Component"): ["DependsOn"],
    ("Component", "Component"): ["HasComponent", "IntegratesWith", "DependsOn"],
    # Fallback for any other relationships
    ("Entity", "Entity"): ["RELATES_TO"],
}


async def add_hierarchical_episode(
    name: str, content: str, source_description: str, reference_time: datetime = None
):
    """
    Add an episode to Graphiti with custom entity and edge types.

    Args:
        name: Name of the episode
        content: Content/body of the episode
        source_description: Description of the source
        reference_time: Reference timestamp
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    try:
        await graphiti.add_episode(
            name=name,
            episode_body=content,
            source_description=source_description,
            reference_time=reference_time,
            entity_types=entity_types,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
        )
        print(f"✅ Added episode: {name}")
        return True
    except Exception as e:
        print(f"❌ Failed to add episode '{name}': {str(e)}")
        return False


def generate_episode_body(content: Content) -> str:
    """
    Convert structured Content into natural language for Graphiti to process.
    Args:
        Content: Structured content object
    Returns:
        Natural Language Description


    """
    episode_text = f"""
    {content.core} is the core organization.
    
    The {content.domain} domain exists within {content.core}.
    
    Within the {content.domain} domain, there is a project called {content.project}.
    
    The {content.project} project has a component named {content.component}.
    
    The {content.component} component includes a feature called {content.feature}.
    The {content.feature} feature {content.description}.

    """
    return episode_text.strip()


async def add_episodes_to_graphiti(
    core: str, domain: str, project: str, component: str, feature: str, description: str
) -> bool:
    """
    This function adds an episode to graph database using graphiti.

    Args:
        core: The knowledge graph root.
        domain: Which domain of the root.
        project: Tell about the Project. Which project.
        component: Which component of the project.
        feature: Which feature of the project we talking.
        description: description of the content.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert parameters to dict for JSON serialization
        name = f"{core} → {domain} → {project} → {component} → {feature}"
        # body
        episode_body = f"""
    {core} is the core organization.
    
    The {domain} domain exists within {core}.
    
    Within the {domain} domain, there is a project called {project}.
    
    The {project} project has a component named {component}.
    
    The {component} component includes a feature called {feature}.
    The {feature} feature {description}.

    """
        # Dynamic description
        source_description = f"{domain} - {component}: {feature}"

        await graphiti.add_episode(
            name=name,
            episode_body=episode_body,
            source=EpisodeType.json,
            source_description=source_description,
            reference_time=datetime.now(timezone.utc),
            edge_types=edge_types,
            entity_types=entity_types,
            edge_type_map=edge_type_map,
        )

        logger.info(f"Successfully added episode: {name}")
        return True

    except Exception as e:
        logger.error(f"Failed to add episode '{name}': {str(e)}")
        return False


store_to_graphiti_tool = StructuredTool.from_function(
    func=add_episodes_to_graphiti,
    name="store_to_graphiti",
    description="This tool is used to add content in the Knowledge Graph",
    return_direct=True,
    args_schema=Content,
    coroutine=add_episodes_to_graphiti,
)


if __name__ == "__main__":

    async def main():
        args = sys.argv
        if len(args) > 1:
            if args[1] == "true":
                # Initialize indices and constraints first
                print("Initializing Neo4j indices and constraints...")
                await initialize_indicies_and_constraints()
                print("Indices initialized successfully!")

        contents = [
            # E-Commerce Platform - MCP Domain
            Content(
                core="Betafit",
                domain="MCP",
                project="E-Commerce Platform",
                component="Payment Service",
                feature="Stripe Integration",
                description="Handles payment processing via Stripe API",
            ),
            Content(
                core="Betafit",
                domain="MCP",
                project="E-Commerce Platform",
                component="Payment Service",
                feature="PayPal Integration",
                description="Handles payment processing via PayPal",
            ),
            Content(
                core="Betafit",
                domain="MCP",
                project="E-Commerce Platform",
                component="Inventory Service",
                feature="Stock Sync",
                description="Synchronizes inventory levels across multiple warehouses",
            ),
            Content(
                core="Betafit",
                domain="MCP",
                project="E-Commerce Platform",
                component="Recommendation Engine",
                feature="Collaborative Filtering",
                description="Recommends products based on user purchase history and trends",
            ),
            Content(
                core="Betafit",
                domain="MCP",
                project="E-Commerce Platform",
                component="Search Service",
                feature="Elasticsearch Integration",
                description="Provides full-text search and filtering for catalog items",
            ),
            Content(
                core="Betafit",
                domain="MCP",
                project="E-Commerce Platform",
                component="Notification Service",
                feature="Email & SMS Alerts",
                description="Sends order confirmations and delivery updates to users",
            ),
            # AI Platform - PCF Parser Domain
            Content(
                core="Betafit",
                domain="PCF Parser",
                project="AI Platform",
                component="Model Training Pipeline",
                feature="Distributed Training",
                description="Trains transformer models across multiple GPUs using PyTorch DDP",
            ),
            Content(
                core="Betafit",
                domain="PCF Parser",
                project="AI Platform",
                component="Inference API",
                feature="Real-time Model Serving",
                description="Provides low-latency inference for LLM models using FastAPI",
            ),
            Content(
                core="Betafit",
                domain="PCF Parser",
                project="AI Platform",
                component="Data Preprocessing",
                feature="Data Normalization",
                description="Cleans and normalizes raw data before feeding into models",
            ),
        ]
        for content in contents:
            result = await store_to_graphiti_tool.ainvoke(input=content.model_dump())
            print(result)
            await asyncio.sleep(30)

        await graphiti.close()
        print("\nConnection closed")

    asyncio.run(main())
