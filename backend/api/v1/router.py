"""Aggregates all v1 routers."""

from fastapi import APIRouter

from backend.api.v1 import (
    train, predict, players, attention, ensemble, games,
    evaluation, data, tuning, schedule, predictions_hub,
    teams, scheduler_routes, baseline, pitchers, accuracy, system,
)

router = APIRouter(prefix="/v1")

router.include_router(train.router)
router.include_router(predict.router)
router.include_router(players.router)
router.include_router(attention.router)
router.include_router(ensemble.router)
router.include_router(games.router)
router.include_router(evaluation.router)
router.include_router(data.router)
router.include_router(tuning.router)
router.include_router(schedule.router)
router.include_router(predictions_hub.router)
router.include_router(teams.router)
router.include_router(scheduler_routes.router)
router.include_router(baseline.router)
router.include_router(pitchers.router)
router.include_router(accuracy.router)
router.include_router(system.router)
