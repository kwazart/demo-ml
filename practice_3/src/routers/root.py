from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def root():
    return {"message": "Hi there! It's a root page."}
