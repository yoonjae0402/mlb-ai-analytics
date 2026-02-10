"""Team endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.db.session import get_db
from backend.db.models import Team
from backend.api.v1.schemas import TeamResponse

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("/", response_model=list[TeamResponse])
async def list_teams(db: AsyncSession = Depends(get_db)):
    """List all MLB teams."""
    result = await db.execute(select(Team).order_by(Team.name))
    teams = result.scalars().all()
    return [
        TeamResponse(
            id=t.id, mlb_id=t.mlb_id, name=t.name,
            abbreviation=t.abbreviation,
            league=t.league, division=t.division,
            logo_url=t.logo_url,
        )
        for t in teams
    ]
