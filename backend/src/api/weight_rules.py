"""
Weight rules management API endpoints
"""
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.orm import Session
import logging

from ..db import get_session
from ..models.schemas import (
    WeightRule, WeightRuleRequest, WeightRuleResponse,
    WeightSimulationRequest, WeightSimulationResponse,
    WeightCalculation, Document
)
from ..services.weight_service import get_weight_service
from ..services.storage import get_storage_service

router = APIRouter(prefix="/api/weight-rules", tags=["weight-rules"])
logger = logging.getLogger(__name__)


@router.get("", response_model=List[WeightRule])
async def get_weight_rules(
    enabled_only: bool = False,
    db: Session = Depends(get_session)
):
    """Get all weight rules"""
    try:
        weight_service = get_weight_service()
        rules = await weight_service.get_active_rules()
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        return rules
    except Exception as e:
        logger.error(f"Error fetching weight rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{rule_id}", response_model=WeightRule)
async def get_weight_rule(
    rule_id: str,
    db: Session = Depends(get_session)
):
    """Get a specific weight rule by ID"""
    try:
        weight_service = get_weight_service()
        rules = await weight_service.get_active_rules()
        
        for rule in rules:
            if rule.id == rule_id:
                return rule
        
        raise HTTPException(status_code=404, detail="Weight rule not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching weight rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=WeightRuleResponse)
async def create_weight_rule(
    request: WeightRuleRequest,
    db: Session = Depends(get_session)
):
    """Create a new weight rule"""
    try:
        # Create the rule
        rule = WeightRule(
            name=request.name,
            rule_type=request.rule_type,
            enabled=request.enabled,
            priority=request.priority,
            conditions=request.conditions,
            weight_modifier=request.weight_modifier
        )
        
        # Get affected documents
        storage = get_storage_service()
        documents = await storage.get_documents()
        
        # Convert to Document models
        doc_models = []
        for doc in documents:
            try:
                doc_models.append(Document(**doc))
            except:
                pass
        
        # Calculate affected count
        weight_service = get_weight_service()
        affected_docs = []
        preview_calculations = []
        
        for doc in doc_models[:10]:  # Preview first 10 documents
            calc = await weight_service.calculate_document_weight(doc, [rule])
            if calc.applied_rules:
                affected_docs.append(doc)
                preview_calculations.append(calc)
        
        rule.affected_count = len(affected_docs)
        
        # TODO: Save rule to database
        
        return WeightRuleResponse(
            rule=rule,
            affected_documents=affected_docs,
            preview_calculations=preview_calculations
        )
    except Exception as e:
        logger.error(f"Error creating weight rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{rule_id}", response_model=WeightRule)
async def update_weight_rule(
    rule_id: str,
    request: WeightRuleRequest,
    db: Session = Depends(get_session)
):
    """Update an existing weight rule"""
    try:
        # TODO: Update rule in database
        
        rule = WeightRule(
            id=rule_id,
            name=request.name,
            rule_type=request.rule_type,
            enabled=request.enabled,
            priority=request.priority,
            conditions=request.conditions,
            weight_modifier=request.weight_modifier
        )
        
        return rule
    except Exception as e:
        logger.error(f"Error updating weight rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{rule_id}")
async def delete_weight_rule(
    rule_id: str,
    db: Session = Depends(get_session)
):
    """Delete a weight rule"""
    try:
        # TODO: Delete rule from database
        
        return {"status": "success", "message": f"Rule {rule_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting weight rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate", response_model=Dict)
async def simulate_weight_rules(
    request: WeightSimulationRequest,
    db: Session = Depends(get_session)
):
    """Simulate the effect of weight rules on documents"""
    try:
        storage = get_storage_service()
        weight_service = get_weight_service()
        
        # Get documents
        all_documents = await storage.get_documents()
        
        # Filter documents if specific IDs provided
        if request.document_ids:
            documents = [d for d in all_documents if d["id"] in request.document_ids]
        else:
            documents = all_documents
        
        # Convert to Document models
        doc_models = []
        for doc in documents:
            try:
                doc_models.append(Document(**doc))
            except:
                pass
        
        # Run simulation
        simulation_result = await weight_service.simulate_rules(
            doc_models,
            request.rules
        )
        
        return simulation_result
    except Exception as e:
        logger.error(f"Error simulating weight rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/{document_id}/calculation", response_model=WeightCalculation)
async def get_document_weight_calculation(
    document_id: str,
    db: Session = Depends(get_session)
):
    """Get the weight calculation breakdown for a specific document"""
    try:
        storage = get_storage_service()
        weight_service = get_weight_service()
        
        # Get document
        doc_data = await storage.get_document(document_id)
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Convert to Document model
        doc = Document(**doc_data)
        
        # Calculate weight with all active rules
        calculation = await weight_service.calculate_document_weight(doc)
        
        return calculation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating document weight for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distribution")
async def get_weight_distribution(
    db: Session = Depends(get_session)
):
    """Get the current weight distribution across all documents"""
    try:
        storage = get_storage_service()
        weight_service = get_weight_service()
        
        # Get all documents
        documents = await storage.get_documents()
        
        # Convert to Document models
        doc_models = []
        for doc in documents:
            try:
                doc_models.append(Document(**doc))
            except:
                pass
        
        # Calculate distribution
        distribution = await weight_service.calculate_weight_distribution(doc_models)
        
        return distribution
    except Exception as e:
        logger.error(f"Error calculating weight distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-update")
async def batch_update_document_weights(
    updates: Dict[str, float] = Body(..., description="Map of document_id to new weight"),
    db: Session = Depends(get_session)
):
    """Batch update document weights"""
    try:
        storage = get_storage_service()
        updated_count = 0
        
        for doc_id, new_weight in updates.items():
            # Validate weight range
            if new_weight < 0.1 or new_weight > 10.0:
                continue
            
            # Get document
            doc = await storage.get_document(doc_id)
            if doc:
                doc["weight"] = new_weight
                await storage.store_document(doc)
                updated_count += 1
        
        return {
            "status": "success",
            "updated_count": updated_count,
            "total_requested": len(updates)
        }
    except Exception as e:
        logger.error(f"Error batch updating document weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))