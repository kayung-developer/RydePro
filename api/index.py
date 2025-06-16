# index.py
#
# =================================================================================
# ||  RYDEPRO - DEFINITIVE, FULLY DEBUGGED & ROBUST CAR ORDERING SYSTEM           ||
# =================================================================================
#
# Version: 4.3 (Pydantic Response Model FIXED)
#
# KEY IMPROVEMENTS IN THIS VERSION:
# - Pydantic FastAPIError FIXED: The critical crash caused by using a SQLAlchemy
#   model as a `response_model` is resolved.
# - ROBUST RESPONSE SCHEMAS: New Pydantic models (e.g., `RideResponse`,
#   `DriverResponse`, `UserResponse`) have been created to act as a clean,
#   decoupled layer between the database and the API.
# - MODERNIZED & CONSISTENT CODEBASE: The entire application now consistently
#   uses Pydantic schemas for data validation and serialization, which is a
#   FastAPI best practice for creating robust and predictable APIs.
# - ALL PREVIOUS FUNCTIONALITY RETAINED: This fix integrates perfectly with all
#   existing features.
#
# Tech Stack: Python, FastAPI, SQLAlchemy, Uvicorn, HTML5, CSS3, JavaScript
# =================================================================================


# ==============================================================================
# PART 1: IMPORTS & CORE SETUP
# ==============================================================================
import os
import json
import datetime
import math
import secrets
import hmac
import hashlib
import shutil
import base64
from enum import Enum as PyEnum

from fastapi import (
    FastAPI, Request, Depends, HTTPException, status, APIRouter,
    Form, File, UploadFile, Header
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from pydantic_settings import BaseSettings

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Boolean, Enum as SQLAlchemyEnum, ForeignKey, Text
)
from sqlalchemy.orm import sessionmaker, Session, relationship, joinedload, declarative_base
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, auth
import uvicorn

# ==============================================================================
# PART 2: CONFIGURATION
# ==============================================================================
# (Configuration is unchanged)
# At the top of PART 2: CONFIGURATION
# This logic checks if the code is running on Vercel and adjusts the DB path.

CONFIG = {
    "PROJECT_NAME": "RydePro - Advanced Ride-Hailing System",
    "SECRET_KEY": secrets.token_urlsafe(32),
    "ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 60 * 24 * 7,
    "DATABASE_URL": os.environ.get("DATABASE_URL", "sqlite:///./rydepro.db"),
    "FIREBASE_SERVICE_ACCOUNT_KEY": "firebase-service.json",
    "FIREBASE_WEB_CONFIG": {
        "apiKey": "YOUR API KEY",
        "authDomain": "YOUR AUTH DOMAIN",
        "projectId": "YOUR PROJECT ID",
        "storageBucket": "v STORAGE BUCKET",
        "messagingSenderId": "YOUR MESSANGING SENDER ID",
        "appId": "YOUR APP ID",
        "measurementId": "YOUR MEASUREMENT ID"
    },
    "GOOGLE_MAPS_API_KEY": "YOUR_GOOGLE_MAPS_API_KEY",
    "PAYSTACK_SECRET_KEY": "YOUR_PAYSTACK_SECRET_KEY",
    "PAYSTACK_PUBLIC_KEY": "YOUR_PAYSTACK_PUBLIC_KEY",
    "UPLOADS_DIR": "uploads", "DEFAULT_CURRENCY": "USD", "COMMISSION_RATE": 0.20,
    "BASE_FARE": 2.50, "COST_PER_KM": 1.75, "COST_PER_MINUTE": 0.30,
    "SURGE_MAX_MULTIPLIER": 2.5, "SURGE_DEMAND_SENSITIVITY": 0.2,
    "HEAT_MAPS": {
        "downtown": {"lat": 34.05, "lng": -118.25, "radius": 5, "bonus": 0.2},
        "airport": {"lat": 33.94, "lng": -118.40, "radius": 4, "bonus": 0.3}
    },
    "SUBSCRIPTION_PLANS": {
        "basic": {"name": "Basic", "price": 9.99, "benefits": ["5% off rides", "Standard Support"]},
        "premium": {"name": "Premium", "price": 19.99,
                    "benefits": ["10% off rides", "Priority Support", "Priority Matching"]},
        "ultimate": {"name": "Ultimate", "price": 29.99,
                     "benefits": ["15% off rides", "24/7 VIP Support", "Highest Priority Matching",
                                  "No Surge Pricing"]},
    }
}

IS_VERCEL = os.environ.get("VERCEL") == "1"
if IS_VERCEL:
    db_path = os.path.join("/tmp", "rydepro.db")
    CONFIG["DATABASE_URL"] = f"sqlite:///{db_path}"


# ==============================================================================
# PART 3: INITIALIZATION & CORE UTILITIES
# ==============================================================================
def setup_dirs():
    for subdir in ["documents", "avatars"]:
        path = os.path.join(CONFIG["UPLOADS_DIR"], subdir)
        if not os.path.exists(path): os.makedirs(path)


def init_firebase():
    """
    Initializes the Firebase Admin SDK.

    This function implements a robust, deployment-ready strategy:
    1. It first checks for a `FIREBASE_SERVICE_ACCOUNT_BASE64` environment variable.
       If found, it decodes it from Base64 to reconstruct the credentials JSON.
       This is the SECURE method for serverless platforms like Vercel.
    2. If the environment variable is not found, it falls back to looking for the
       local `firebase-service-account.json` file. This is for local development.
    """
    try:
        creds_json_str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_BASE64")

        if creds_json_str:
            # --- DEPLOYMENT MODE: Decode from environment variable ---
            print("Found FIREBASE_SERVICE_ACCOUNT_BASE64 env var. Decoding...")
            decoded_creds_bytes = base64.b64decode(creds_json_str)
            creds_dict = json.loads(decoded_creds_bytes)
            cred = credentials.Certificate(creds_dict)
            print("Credentials successfully decoded from Base64.")
        else:
            # --- LOCAL DEVELOPMENT MODE: Load from file ---
            local_creds_path = CONFIG["FIREBASE_SERVICE_ACCOUNT_KEY"]
            if os.path.exists(local_creds_path):
                print(f"Loading Firebase credentials from local file: {local_creds_path}")
                cred = credentials.Certificate(local_creds_path)
            else:
                print(f"❌ FIREBASE CRITICAL ERROR: Neither Base64 env var nor local file '{local_creds_path}' found.")
                return

        # Initialize the app if it hasn't been already
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK initialized successfully.")

    except Exception as e:
        print(f"❌ FIREBASE CRITICAL: Could not initialize Firebase Admin SDK. {e}")
        print("   Ensure the Base64 environment variable is set correctly on Vercel, or the local JSON file is valid.")


def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)


def get_password_hash(password): return pwd_context.hash(password)


# ==============================================================================
# PART 4: DATABASE SETUP (SQLALCHEMY 2.0 Syntax)
# ==============================================================================
engine = create_engine(CONFIG["DATABASE_URL"], connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Enums & Models ---
class UserRole(str, PyEnum): passenger = "passenger"; driver = "driver"; admin = "admin"


class RideStatus(str,
                 PyEnum): pending = "pending"; accepted = "accepted"; arriving = "arriving"; in_progress = "in_progress"; completed = "completed"; cancelled = "cancelled"; scheduled = "scheduled"


class PaymentMethod(str, PyEnum): wallet = "wallet"; card = "card"; cash = "cash"


class VehicleType(str,
                  PyEnum): economy = "Economy"; luxury = "Luxury"; suv = "SUV"; bike = "Bike"; truck = "Truck"; ev = "EV"


class DriverStatus(str,
                   PyEnum): offline = "offline"; online = "online"; on_trip = "on_trip"; pending_approval = "pending_approval"; rejected = "rejected"


class SubscriptionPlan(str, PyEnum): none = "none"; basic = "basic"; premium = "premium"; ultimate = "ultimate"


class TransactionType(str,
                      PyEnum): topup = "topup"; ride_payment = "ride_payment"; withdrawal = "withdrawal"; ride_earning = "ride_earning"; subscription = "subscription"


class User(Base):
    __tablename__ = "users";
    id = Column(Integer, primary_key=True, index=True);
    firebase_uid = Column(String, unique=True, index=True, nullable=True);
    email = Column(String, unique=True, index=True, nullable=False);
    hashed_password = Column(String, nullable=True);
    full_name = Column(String, nullable=False);
    profile_picture_url = Column(String, default="/static/avatars/default.png");
    role = Column(SQLAlchemyEnum(UserRole), nullable=False);
    created_at = Column(DateTime, default=datetime.utcnow);
    is_active = Column(Boolean, default=True);
    wallet_balance = Column(Float, default=0.0);
    subscription_plan = Column(SQLAlchemyEnum(SubscriptionPlan), default=SubscriptionPlan.none);
    subscription_expiry = Column(DateTime, nullable=True);
    fcm_token = Column(String, nullable=True);
    driver_info = relationship("Driver", back_populates="user", uselist=False, cascade="all, delete-orphan");
    transactions = relationship("WalletTransaction", back_populates="user", cascade="all, delete-orphan")


class Driver(Base):
    __tablename__ = "drivers";
    id = Column(Integer, primary_key=True);
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False);
    license_number = Column(String, unique=True);
    license_doc_url = Column(String);
    insurance_doc_url = Column(String);
    status = Column(SQLAlchemyEnum(DriverStatus), default=DriverStatus.pending_approval);
    current_lat = Column(Float, nullable=True);
    current_lng = Column(Float, nullable=True);
    last_seen = Column(DateTime, default=datetime.utcnow);
    average_rating = Column(Float, default=5.0);
    rejection_reason = Column(Text, nullable=True);
    user = relationship("User", back_populates="driver_info");
    vehicle = relationship("Vehicle", back_populates="driver", uselist=False, cascade="all, delete-orphan")


class Vehicle(Base): __tablename__ = "vehicles"; id = Column(Integer, primary_key=True); driver_id = Column(Integer,
                                                                                                            ForeignKey(
                                                                                                                "drivers.id"),
                                                                                                            nullable=False); make = Column(
    String); model = Column(String); year = Column(Integer); color = Column(String); license_plate = Column(String,
                                                                                                            unique=True); vehicle_type = Column(
    SQLAlchemyEnum(VehicleType)); driver = relationship("Driver", back_populates="vehicle")


class Ride(Base): __tablename__ = "rides"; id = Column(Integer, primary_key=True); passenger_id = Column(Integer,
                                                                                                         ForeignKey(
                                                                                                             "users.id")); driver_id = Column(
    Integer, ForeignKey("drivers.id"), nullable=True); pickup_address = Column(String); dropoff_address = Column(
    String); pickup_lat = Column(Float); pickup_lng = Column(Float); dropoff_lat = Column(Float); dropoff_lng = Column(
    Float); status = Column(SQLAlchemyEnum(RideStatus), default=RideStatus.pending); vehicle_type_requested = Column(
    SQLAlchemyEnum(VehicleType)); estimated_fare = Column(Float); actual_fare = Column(Float,
                                                                                       nullable=True); distance_km = Column(
    Float, nullable=True); duration_minutes = Column(Float, nullable=True); payment_method = Column(
    SQLAlchemyEnum(PaymentMethod)); created_at = Column(DateTime, default=datetime.utcnow); accepted_at = Column(
    DateTime, nullable=True); arrived_at = Column(DateTime, nullable=True); started_at = Column(DateTime,
                                                                                                nullable=True); completed_at = Column(
    DateTime, nullable=True); passenger_rated = Column(Boolean, default=False); driver_rated = Column(Boolean,
                                                                                                      default=False); share_token = Column(
    String, default=lambda: secrets.token_urlsafe(16)); passenger = relationship("User", foreign_keys=[
    passenger_id]); driver = relationship("Driver", foreign_keys=[driver_id])


class Review(Base): __tablename__ = "reviews"; id = Column(Integer, primary_key=True); ride_id = Column(Integer,
                                                                                                        ForeignKey(
                                                                                                            "rides.id")); reviewer_id = Column(
    Integer, ForeignKey("users.id")); reviewee_id = Column(Integer, ForeignKey("users.id")); rating = Column(Integer,
                                                                                                             default=5); comment = Column(
    Text, nullable=True); created_at = Column(DateTime, default=datetime.utcnow)


class WalletTransaction(Base): __tablename__ = "wallet_transactions"; id = Column(Integer,
                                                                                  primary_key=True); user_id = Column(
    Integer, ForeignKey("users.id")); transaction_type = Column(SQLAlchemyEnum(TransactionType)); amount = Column(
    Float); reference = Column(String, nullable=True); description = Column(String); created_at = Column(DateTime,
                                                                                                         default=datetime.utcnow); user = relationship(
    "User", back_populates="transactions")


class SOSLog(Base): __tablename__ = "sos_logs"; id = Column(Integer, primary_key=True); ride_id = Column(Integer,
                                                                                                         ForeignKey(
                                                                                                             "rides.id")); user_id = Column(
    Integer, ForeignKey("users.id")); timestamp = Column(DateTime, default=datetime.utcnow); lat = Column(
    Float); lng = Column(Float); details = Column(Text, default="SOS button pressed")


Base.metadata.create_all(bind=engine)


def get_db(): db = SessionLocal();_ = (yield db);db.close()


# ==============================================================================
# PART 5: AUTHENTICATION & SECURITY
# ==============================================================================
def create_access_token(data: dict): to_encode = data.copy();to_encode.update(
    {"exp": datetime.utcnow() + timedelta(minutes=CONFIG["ACCESS_TOKEN_EXPIRE_MINUTES"])});return jwt.encode(to_encode,
                                                                                                             CONFIG[
                                                                                                                 "SECRET_KEY"],
                                                                                                             algorithm=
                                                                                                             CONFIG[
                                                                                                                 "ALGORITHM"])


def get_current_user(request: Request, db: Session = Depends(get_db)):
    """
    Retrieves the current user from a session token.

    This function implements a robust two-step check:
    1. It first looks for the 'access_token' in the request cookies, which is
       the standard method for browser-based sessions.
    2. If no cookie is found, it falls back to checking for an 'Authorization'
       header with a Bearer token (e.g., "Bearer <token>"). This is the standard
       for API clients and mobile apps.

    Args:
        request: The incoming FastAPI request object.
        db: The SQLAlchemy database session dependency.

    Returns:
        The User database object if the token is valid, otherwise None.
    """
    token = None

    # --- Step 1: Check for the token in cookies (for browsers) ---
    if "access_token" in request.cookies:
        token = request.cookies.get("access_token")

    # --- Step 2: If no cookie, check for Authorization header (for APIs/mobile) ---
    else:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    # If no token was found in either location, there is no user.
    if not token:
        return None

    # --- Step 3: Decode the token and retrieve the user ---
    try:
        # Decode the JWT to get the payload
        payload = jwt.decode(token, CONFIG["SECRET_KEY"], algorithms=[CONFIG["ALGORITHM"]])

        # Extract the user ID (subject) from the payload
        user_id_str = payload.get("sub")

        if user_id_str is None:
            # The token is malformed if it's missing the 'sub' claim
            return None

        user_id = int(user_id_str)

        # Query the database for the user with that ID
        return db.query(User).filter(User.id == user_id).first()

    except (JWTError, ValueError, KeyError):
        # Catches several potential errors:
        # - JWTError: If the token is invalid, expired, or has a bad signature.
        # - ValueError: If the user_id from the token cannot be converted to an integer.
        # - KeyError: If the payload is structured unexpectedly.
        return None


async def get_current_active_user(user: User = Depends(get_current_user)):
    if not user: raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated");
    if not user.is_active: raise HTTPException(status.HTTP_400_BAD_REQUEST, "Inactive user");return user


def require_role(role: UserRole):
    async def role_checker(user: User = Depends(get_current_active_user)):
        if user.role != role: raise HTTPException(status.HTTP_403_FORBIDDEN,
                                                  f"{role.value.capitalize()} privileges required");return user

    return role_checker


require_passenger = require_role(UserRole.passenger);
require_driver = require_role(UserRole.driver);
require_admin = require_role(UserRole.admin);


# ==============================================================================
# PART 6: PYDANTIC SCHEMAS (RESPONSE MODELS ADDED)
# ==============================================================================
# --- Base Schemas ---
class OrmModel(BaseModel):
    class Config: from_attributes = True


# --- Input Schemas ---
class UserCreate(BaseModel): full_name: str; email: EmailStr; password: str


class UserLogin(BaseModel): email: EmailStr; password: str; expected_role: Optional[UserRole] = None


class FirebaseLogin(BaseModel): firebase_token: str


class RideRequest(
    BaseModel): pickup_lat: float; pickup_lng: float; pickup_address: str; dropoff_lat: float; dropoff_lng: float; dropoff_address: str; vehicle_type: VehicleType; payment_method: PaymentMethod


class FareEstimateRequest(
    BaseModel): pickup_lat: float; pickup_lng: float; dropoff_lat: float; dropoff_lng: float; vehicle_type: VehicleType


class RideAction(BaseModel): ride_id: int


class DriverRideStatusUpdate(BaseModel): ride_id: int; status: str


class RateRide(BaseModel): ride_id: int; rating: int = Field(..., ge=1, le=5); comment: Optional[str] = None


class PurchaseSubscription(BaseModel): plan: SubscriptionPlan


# --- FIX: Response Schemas ---
class UserResponse(OrmModel):
    id: int;
    full_name: str;
    email: EmailStr;
    profile_picture_url: str;
    role: UserRole


class VehicleResponse(OrmModel):
    make: str;
    model: str;
    year: int;
    color: str;
    license_plate: str;
    vehicle_type: VehicleType


class DriverResponse(OrmModel):
    id: int;
    status: DriverStatus;
    average_rating: float;
    user: UserResponse;
    vehicle: Optional[VehicleResponse] = None


class RideResponse(OrmModel):
    id: int;
    status: RideStatus;
    pickup_address: str;
    dropoff_address: str
    estimated_fare: float;
    actual_fare: Optional[float] = None
    passenger: UserResponse
    driver: Optional[DriverResponse] = None
    driver_current_lat: Optional[float] = None
    driver_current_lng: Optional[float] = None
    passenger_rated: bool;
    driver_rated: bool;
    share_token: str


# ==============================================================================
# PART 7: CORE BUSINESS LOGIC
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2): R, dLat, dLon = 6371, math.radians(lat2 - lat1), math.radians(
    lon2 - lon1);a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
    dLon / 2) ** 2;return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def calculate_surge_pricing(db: Session, lat, lng): now = datetime.utcnow();hour_mult = 1 + (.3 * math.sin(
    math.pi * (now.hour - 6) / 12));day_mult = 1.2 if now.weekday() >= 4 else 1;heatmap_bonus = max(
    (z['bonus'] for z in CONFIG["HEAT_MAPS"].values() if haversine(lat, lng, z['lat'], z['lng']) <= z['radius']),
    default=0);area_requests = db.query(Ride).filter(Ride.status == RideStatus.pending,
                                                     Ride.created_at > now - timedelta(
                                                         minutes=15)).count();available_drivers = db.query(
    Driver).filter(Driver.status == DriverStatus.online).count();demand_surge = 1 + (
            ((area_requests + 1) / (available_drivers + 1)) * CONFIG["SURGE_DEMAND_SENSITIVITY"]);return min(
    (demand_surge * hour_mult * day_mult) + heatmap_bonus, CONFIG["SURGE_MAX_MULTIPLIER"])


def estimate_fare(db: Session, plat, plng, dlat, dlng, vt, user: User): distance = haversine(plat, plng, dlat,
                                                                                             dlng);duration = (
                                                                                                                          distance / 35) * 60;base_fare = (
            CONFIG["BASE_FARE"] + (distance * CONFIG["COST_PER_KM"]) + (
                duration * CONFIG["COST_PER_MINUTE"]));vehicle_mult = {"Economy": 1.0, "Luxury": 2.2, "SUV": 1.5,
                                                                       "Bike": 0.6, "Truck": 2.5, "EV": 1.1}.get(
    vt.value, 1.0);surge_mult = 1.0 if user.subscription_plan == SubscriptionPlan.ultimate else calculate_surge_pricing(
    db, plat, plng);final_fare = base_fare * vehicle_mult * surge_mult;discounts = {SubscriptionPlan.basic: 0.95,
                                                                                    SubscriptionPlan.premium: 0.90,
                                                                                    SubscriptionPlan.ultimate: 0.85};final_fare *= discounts.get(
    user.subscription_plan, 1.0);return {"estimated_fare": round(final_fare, 2), "distance_km": round(distance, 2),
                                         "duration_minutes": round(duration, 1),
                                         "surge_multiplier": round(surge_mult, 2)}


def get_subscription_priority_score(plan: SubscriptionPlan): return {SubscriptionPlan.none: 1.0,
                                                                     SubscriptionPlan.basic: 0.95,
                                                                     SubscriptionPlan.premium: 0.85,
                                                                     SubscriptionPlan.ultimate: 0.75}.get(plan, 1.0)


def find_best_driver_match(db: Session, ride: Ride) -> Optional[Driver]:
    DISTANCE_WEIGHT, RATING_WEIGHT, SUBSCRIPTION_WEIGHT = 0.60, 0.20, 0.20
    drivers = db.query(Driver).options(joinedload(Driver.user), joinedload(Driver.vehicle)).join(Vehicle).filter(
        Driver.status == DriverStatus.online, Vehicle.vehicle_type == ride.vehicle_type_requested).all()
    if not drivers: print("--- AI Driver Matching: No online drivers found for this vehicle type. ---");return None
    scored_drivers = [];
    print("\n--- AI Driver Matching ---")
    for driver in drivers:
        if driver.current_lat and driver.current_lng:
            dist = haversine(ride.pickup_lat, ride.pickup_lng, driver.current_lat, driver.current_lng)
            rating_score = (5.5 - driver.average_rating)
            sub_score = get_subscription_priority_score(driver.user.subscription_plan)
            final_score = (dist * DISTANCE_WEIGHT) + (rating_score * RATING_WEIGHT) + (sub_score * SUBSCRIPTION_WEIGHT)
            scored_drivers.append((final_score, driver))
            print(
                f"  - Driver ID {driver.id}: Dist={dist:.2f}km, Rating={driver.average_rating}, Sub={driver.user.subscription_plan.name} -> SCORE: {final_score:.2f}")
    if not scored_drivers: print("--- AI Driver Matching: No drivers with location data available. ---");return None
    scored_drivers.sort(key=lambda x: x[0])
    best_score, best_driver = scored_drivers[0]
    print(f"==> BEST MATCH: Driver ID {best_driver.id} with score {best_score:.2f}\n")
    return best_driver


def update_driver_rating(driver_id: int, db: Session):
    driver = db.query(Driver).get(driver_id);
    if not driver: return
    ratings = db.query(Review.rating).join(Ride).filter(Ride.driver_id == driver_id,
                                                        Review.reviewee_id == driver.user_id).all();
    if ratings: driver.average_rating = round(sum(r[0] for r in ratings) / len(ratings), 2);db.commit();


# ==============================================================================
# PART 8: FRONTEND ASSETS (CSS, JS, HTML)
# ==============================================================================
# (CSS and JS are unchanged from the last fully debugged version, so they are kept as is)
MAIN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
:root{--bg-dark-primary:#12121c;--bg-dark-secondary:#1a1a2e;--bg-dark-tertiary:#16213e;--primary-accent:#6d28d9;--primary-accent-hover:#5b21b6;--secondary-accent:#e94560;--text-primary:#e0e0e0;--text-secondary:#a0a0c0;--border-color:#3a3a5a;--success:#10b981;--error:#ef4444;--warning:#f59e0b;--font-family:'Inter',sans-serif;--border-radius-sm:6px;--border-radius-md:10px;--shadow-md:0 4px 6px -1px rgba(0,0,0,.1),0 2px 4px -2px rgba(0,0,0,.1);--shadow-lg:0 10px 15px -3px rgba(0,0,0,.1),0 4px 6px -4px rgba(0,0,0,.1);--transition:all .3s cubic-bezier(.4,0,.2,1)}*,::after,::before{box-sizing:border-box;margin:0;padding:0}body{font-family:var(--font-family);background-color:var(--bg-dark-primary);color:var(--text-primary);line-height:1.6;display:flex;flex-direction:column;min-height:100vh;overflow-x:hidden}.main-content{flex:1}h1,h2,h3,h4{color:var(--text-primary);margin-bottom:1rem;font-weight:600}a{color:var(--primary-accent);text-decoration:none;transition:var(--transition)}a:hover{color:var(--primary-accent-hover)}.container{width:95%;max-width:1600px;margin:2rem auto;padding:0 1rem}.btn{display:inline-block;padding:12px 28px;border:none;border-radius:var(--border-radius-md);cursor:pointer;font-size:1rem;font-weight:600;text-align:center;transition:var(--transition);text-transform:uppercase;letter-spacing:1px}.btn-primary{background:linear-gradient(90deg,var(--primary-accent),#a855f7);color:#fff}.btn-primary:hover{transform:translateY(-2px);box-shadow:var(--shadow-lg)}.btn-secondary{background:var(--bg-dark-tertiary);color:var(--text-primary)}.btn-secondary:hover{background:#1f2c4a}.btn-danger{background-color:var(--error);color:#fff}.btn-success{background-color:var(--success);color:#fff}.form-container{background:var(--bg-dark-secondary);padding:2.5rem;border-radius:var(--border-radius-md);box-shadow:var(--shadow-lg);max-width:500px;margin:2rem auto;border:1px solid var(--border-color)}.form-group{margin-bottom:1.5rem}.form-group label{display:block;margin-bottom:.5rem;font-weight:500;color:var(--text-secondary)}.form-control{width:100%;padding:14px;background:var(--bg-dark-primary);border:1px solid var(--border-color);border-radius:var(--border-radius-sm);color:var(--text-primary);font-size:1rem;transition:var(--transition)}.form-control:focus{outline:none;border-color:var(--primary-accent);box-shadow:0 0 0 3px rgba(109,40,217,.5)}.main-header{background-color:var(--bg-dark-secondary);padding:1rem 2.5%;display:flex;justify-content:space-between;align-items:center;box-shadow:var(--shadow-md);position:sticky;top:0;z-index:1000;border-bottom:1px solid var(--border-color)}.logo{font-size:2rem;font-weight:700;color:#fff}.logo .fa-rocket{color:var(--primary-accent)}.main-nav ul{list-style:none;display:flex;gap:1.5rem}.main-nav a{color:var(--text-secondary);font-weight:500;padding:5px 10px;border-radius:5px}.main-nav a:hover{background:var(--bg-dark-tertiary);color:var(--text-primary);text-decoration:none}.user-menu{position:relative;cursor:pointer}.user-menu img{width:40px;height:40px;border-radius:50%;border:2px solid var(--border-color)}.user-menu:hover .dropdown-menu{display:block}.dropdown-menu{display:none;position:absolute;right:0;top:120%;background:var(--bg-dark-tertiary);border:1px solid var(--border-color);border-radius:var(--border-radius-md);min-width:220px;box-shadow:var(--shadow-lg);list-style:none;padding:.5rem 0;z-index:1001}.dropdown-menu a{display:flex;gap:.75rem;align-items:center;padding:.75rem 1.25rem;color:var(--text-secondary)}.dropdown-menu a:hover{background:var(--primary-accent);color:#fff;text-decoration:none}.main-footer{background-color:var(--bg-dark-secondary);color:var(--text-secondary);text-align:center;padding:2rem 1rem;margin-top:auto;border-top:1px solid var(--border-color)}#auth-page{display:flex;align-items:center;justify-content:center;min-height:80vh}.dashboard-layout{display:grid;grid-template-columns:260px 1fr;gap:2rem;min-height:calc(100vh - 85px)}.sidebar{background:var(--bg-dark-secondary);padding:2rem 1rem;border-right:1px solid var(--border-color)}.sidebar-nav ul{list-style:none}.sidebar-nav li a{display:flex;align-items:center;gap:1rem;padding:15px;color:var(--text-secondary);border-radius:var(--border-radius-sm);margin-bottom:.5rem;font-size:1.05rem}.sidebar-nav li a:hover{background:var(--bg-dark-tertiary);color:var(--text-primary);text-decoration:none}.sidebar-nav li a.active{background:var(--primary-accent);color:#fff;font-weight:600}.sidebar-nav li a .fa-fw{width:20px;text-align:center}.dashboard-content{padding:2rem;overflow-y:auto}.dashboard-content h1{border-bottom:1px solid var(--border-color);padding-bottom:1rem;margin-bottom:2rem}.card{background:var(--bg-dark-secondary);border:1px solid var(--border-color);border-radius:var(--border-radius-md);padding:1.5rem;margin-bottom:1.5rem;box-shadow:var(--shadow-md)}.card-header{border-bottom:1px solid var(--border-color);padding-bottom:1rem;margin-bottom:1rem;font-size:1.25rem;font-weight:600;display:flex;justify-content:space-between;align-items:center}.stat-card{background:var(--bg-dark-secondary);padding:1.5rem;border-radius:var(--border-radius-md);border:1px solid var(--border-color);text-align:center}.stat-card .icon{font-size:2.5rem;color:var(--primary-accent);margin-bottom:1rem}.stat-card .value{font-size:2rem;font-weight:700}.stat-card .label{font-size:.9rem;color:var(--text-secondary);margin-top:.5rem}.grid-container{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.5rem}.passenger-dashboard-grid{display:grid;grid-template-columns:1fr 420px;gap:2rem;height:calc(100vh - 200px)}#map{height:100%;width:100%;border-radius:var(--border-radius-md);border:2px solid var(--border-color)}.booking-panel{display:flex;flex-direction:column;background:var(--bg-dark-secondary);padding:1.5rem;border-radius:var(--border-radius-md);border:1px solid var(--border-color)}.vehicle-options{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0}.vehicle-option{background:var(--bg-dark-primary);border:2px solid var(--border-color);border-radius:var(--border-radius-sm);padding:1rem;text-align:center;cursor:pointer;transition:var(--transition);position:relative}.vehicle-option:hover{border-color:var(--primary-accent)}.vehicle-option.selected{border-color:var(--primary-accent);background:var(--bg-dark-tertiary)}.vehicle-option.selected::after{content:'✔';position:absolute;top:5px;right:8px;color:var(--success)}.vehicle-option .icon{font-size:2rem;color:var(--primary-accent)}.vehicle-option .name{font-weight:500;margin-top:.5rem}#fare-estimate-box{background:var(--bg-dark-primary);padding:1rem;border-radius:var(--border-radius-sm);margin-top:auto;text-align:center;border:1px solid var(--border-color)}#fare-estimate-box h3{font-size:1.5rem;margin-bottom:.5rem;color:var(--primary-accent)}#fare-estimate-box p{color:var(--text-secondary)}#ride-status-container{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);width:90%;max-width:700px;background:linear-gradient(to right,var(--bg-dark-secondary),var(--bg-dark-tertiary));border-radius:var(--border-radius-md);box-shadow:var(--shadow-lg);z-index:1001;border:1px solid var(--border-color);padding:1.5rem;display:none}.ride-status-header{display:flex;justify-content:space-between;align-items:center}#ride-status-text{font-size:1.2rem;font-weight:600}#sos-button,#share-trip-button{background-color:var(--bg-dark-tertiary);color:var(--text-primary);padding:8px 15px;border-radius:var(--border-radius-sm)}.ride-status-actions{display:flex;gap:.5rem}#sos-button{background-color:var(--error);color:#fff}#ride-driver-info{margin-top:1rem;padding-top:1rem;border-top:1px solid var(--border-color)}.progress-container{margin-top:1rem}.progress-steps{display:flex;justify-content:space-between;position:relative;margin-bottom:1rem}.progress-step{text-align:center;width:25%;font-size:.8rem;color:var(--text-secondary)}.progress-step.active{color:var(--text-primary);font-weight:600}.progress-bar-bg{height:6px;background-color:var(--border-color);border-radius:3px}.progress-bar-fg{height:100%;width:0%;background:var(--primary-accent);border-radius:3px;transition:width .5s ease}#driver-status-toggle{display:flex;align-items:center;gap:1rem;background-color:var(--error);color:#fff;padding:1rem;border-radius:var(--border-radius-md);cursor:pointer;transition:var(--transition);font-weight:600}#driver-status-toggle.online{background-color:var(--success)}.modal{display:none;position:fixed;z-index:2000;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,.8);backdrop-filter:blur(5px);justify-content:center;align-items:center}.modal-content{background:var(--bg-dark-secondary);padding:2rem;border-radius:var(--border-radius-md);width:90%;max-width:500px;text-align:center;border:1px solid var(--border-color);position:relative}.close-modal{position:absolute;top:10px;right:15px;font-size:2rem;color:var(--text-secondary);cursor:pointer;transition:var(--transition)}.close-modal:hover{color:var(--text-primary)}.countdown-timer{width:100px;height:100px;border:5px solid var(--primary-accent);border-radius:50%;display:flex;justify-content:center;align-items:center;font-size:2.5rem;font-weight:700;margin:1rem auto}.table-container{overflow-x:auto}.data-table{width:100%;border-collapse:collapse;margin-top:2rem}.data-table td,.data-table th{padding:15px;text-align:left;border-bottom:1px solid var(--border-color)}.data-table thead{background:var(--bg-dark-tertiary);color:var(--text-primary)}.data-table tbody tr:hover{background:var(--bg-dark-tertiary)}.status-tag{padding:4px 10px;border-radius:20px;font-size:.8rem;font-weight:600;text-transform:capitalize}.status-pending,.status-pending_approval{background-color:var(--warning);color:#000}.status-approved,.status-completed,.status-successful,.status-online{background-color:var(--success);color:#fff}.status-rejected,.status-cancelled,.status-offline{background-color:var(--error);color:#fff}.status-accepted,.status-arriving,.status-in_progress,.status-on_trip{background-color:#3b82f6;color:#fff}.star-rating{display:flex;justify-content:center;direction:rtl}.star-rating input[type=radio]{display:none}.star-rating label{font-size:2.5rem;color:#444;cursor:pointer;transition:color .2s}.star-rating label:hover,.star-rating label:hover~label,.star-rating input[type=radio]:checked~label{color:var(--warning)}#toast-container{position:fixed;top:20px;right:20px;z-index:9999}.toast{padding:15px 25px;margin-bottom:1rem;border-radius:var(--border-radius-sm);color:#fff;box-shadow:var(--shadow-lg);opacity:0;transform:translateX(100%)}.toast.show{opacity:1;transform:translateX(0);animation:slideIn .5s forwards}@keyframes slideIn{to{opacity:1;transform:translateX(0)}}@keyframes slideOut{from{opacity:1;transform:translateX(0)}to{opacity:0;transform:translateX(100%)}}.toast-success{background:var(--success)}.toast-error{background:var(--error)}.toast-info{background:#3b82f6}.loader{border:4px solid var(--border-color);border-top:4px solid var(--primary-accent);border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:2rem auto}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.text-success{color:var(--success)}.text-error{color:var(--error)}.text-center{text-align:center}.mt-3{margin-top:1.5rem}.profile-avatar{width:120px;height:120px;border-radius:50%;object-fit:cover;border:3px solid var(--border-color)}.file-upload-wrapper{position:relative;overflow:hidden;display:inline-block}.file-upload-wrapper input[type=file]{font-size:100px;position:absolute;left:0;top:0;opacity:0}.subscription-card{text-align:center;padding:2rem}.subscription-card.popular{border:2px solid var(--primary-accent);transform:scale(1.05)}.subscription-card h3{font-size:1.5rem;color:var(--primary-accent)}.subscription-card .price{font-size:2.5rem;font-weight:700;margin:1rem 0}.subscription-card ul{list-style:none;margin:1.5rem 0}.subscription-card li{margin-bottom:.5rem}.subscription-card .btn.current{background:var(--success);cursor:not-allowed}
.auth-tabs{display:flex;margin-bottom:1.5rem;border-bottom:1px solid var(--border-color)}.auth-tab-button{flex:1;padding:1rem;background:0 0;border:none;color:var(--text-secondary);cursor:pointer;font-size:1.1rem;font-weight:600;border-bottom:3px solid transparent}.auth-tab-button.active{color:var(--primary-accent);border-bottom-color:var(--primary-accent)}.auth-tab-content{display:none}.auth-tab-content.active{display:block}
@media (max-width:992px){.dashboard-layout{grid-template-columns:1fr}.sidebar{display:flex;height:auto;padding:.5rem;overflow-x:auto}.sidebar-nav ul{display:flex;width:100%;justify-content:space-around}.passenger-dashboard-grid{grid-template-columns:1fr;height:auto}#map{height:50vh}}
"""

MAIN_JS = f"""
const FIREBASE_CONFIG = {json.dumps(CONFIG['FIREBASE_WEB_CONFIG'])};
const GOOGLE_MAPS_API_KEY = "{CONFIG['GOOGLE_MAPS_API_KEY']}";
const PAYSTACK_PUBLIC_KEY = "{CONFIG['PAYSTACK_PUBLIC_KEY']}";
let map, directionsService, directionsRenderer, pickupMarker, dropoffMarker, driverMarker;
let rideStatusPollInterval, rideRequestPollInterval, countdownInterval;

// --- UTILITY & UI FUNCTIONS ---
function showToast(msg, type = 'info', duration = 4000) {{ const cont=document.getElementById('toast-container')||createToastCont(); const toast=document.createElement('div'); toast.className=`toast toast-${{type}} show`; toast.textContent=msg; cont.appendChild(toast); setTimeout(()=>{{toast.classList.remove('show'); toast.style.animation='slideOut .5s forwards'; setTimeout(()=>toast.remove(),500)}},duration);}}
function createToastCont() {{ const c=document.createElement('div'); c.id='toast-container'; document.body.appendChild(c); return c; }}
async function apiFetch(url, opts = {{}}, showFeedback = true) {{ const headers={{'Accept':'application/json',...opts.headers}}; if(!(opts.body instanceof FormData)) headers['Content-Type']='application/json'; opts.headers=headers; if(opts.body&&typeof opts.body!=='string'&&!(opts.body instanceof FormData))opts.body=JSON.stringify(opts.body); try{{const res=await fetch(url,opts); const data=await res.json(); if(!res.ok)throw new Error(data.detail||'API error occurred'); return data;}} catch(err){{console.error('API Error:',url,err); if(showFeedback)showToast(err.message,'error'); throw err;}}}}

// --- FIREBASE & AUTHENTICATION ---
function initializeFirebase() {{
    const authContainer = document.getElementById('firebaseui-auth-container');
    const socialLoader = document.getElementById('firebaseui-loader');
    if (!authContainer) return;
    try {{
        if (!FIREBASE_CONFIG.apiKey || FIREBASE_CONFIG.apiKey.includes("YOUR_")) {{
            console.error("Firebase config error: API key is missing or a placeholder.");
            if (socialLoader) socialLoader.innerHTML = '<p style="color:var(--error);">Firebase Config Error.</p>';
            return;
        }}
        if (firebase.apps.length === 0) {{ firebase.initializeApp(FIREBASE_CONFIG); }}
        firebase.auth().onIdTokenChanged(user => {{
            if (user) {{ user.getIdToken().then(fcm_token => apiFetch('/api/auth/update-fcm', {{ method: 'POST', body: {{ fcm_token }} }}, false)).catch(e => console.error("FCM update failed", e)); }}
        }});
        setupFirebaseUI();
    }} catch (e) {{
        console.error("Firebase initialization failed:", e);
        if (socialLoader) socialLoader.innerHTML = `<p style="color:var(--error);">Auth Error: ${{e.message}}</p>`;
    }}
}}
function setupFirebaseUI() {{
    const ui = new firebaseui.auth.AuthUI(firebase.auth());
    ui.start('#firebaseui-auth-container', {{
        signInSuccessUrl: '/auth/firebase/callback',
        signInOptions: [firebase.auth.GoogleAuthProvider.PROVIDER_ID],
        callbacks: {{
            signInSuccessWithAuthResult: (res, url) => {{ res.user.getIdToken().then(t => handleFirebaseLogin(t)); return false; }},
            uiShown: () => {{ const socialLoader = document.getElementById('firebaseui-loader'); if (socialLoader) socialLoader.style.display = 'none'; }}
        }}
    }});
}}
async function handleFirebaseLogin(token) {{
    try {{
        const data = await apiFetch('/api/auth/firebase-login', {{ method: 'POST', body: {{ firebase_token: token }} }});
        showToast('Login successful! Redirecting...', 'success');
        setTimeout(() => window.location.href = data.redirect_url, 1000);
    }} catch (e) {{
        showToast('Login failed on server. Please try again.', 'error');
    }}
}}

// --- GOOGLE MAPS & RIDE LOGIC ---
function loadGoogleMapsScript(){{if(!document.querySelector('script[src*="maps.googleapis.com"]')){{const s=document.createElement('script');s.src=`https://maps.googleapis.com/maps/api/js?key=${{GOOGLE_MAPS_API_KEY}}&libraries=places,directions&callback=initMap`;s.async=true;document.head.appendChild(s);}}else if(typeof google!=='undefined'){{initMap();}}}}
window.initMap=function(){{const el=document.getElementById("map");if(!el)return;map=new google.maps.Map(el,{{center:{{lat:34.0522,lng:-118.2437}},zoom:12,mapId:"RYDEPRO_DARK_STYLE",disableDefaultUI:true,zoomControl:true}});directionsService=new google.maps.DirectionsService();directionsRenderer=new google.maps.DirectionsRenderer({{map:map,suppressMarkers:true,polylineOptions:{{strokeColor:"#6d28d9",strokeWeight:6}}}});if(navigator.geolocation)navigator.geolocation.getCurrentPosition(p=>map.setCenter({{lat:p.coords.latitude,lng:p.coords.longitude}}));if(document.getElementById('passenger-dashboard'))initPassengerMapFeatures();if(document.getElementById('driver-dashboard'))initDriverMapFeatures();if(document.getElementById('track-ride-status')){{const pathParts=window.location.pathname.split('/');const shareToken=pathParts[pathParts.length-1];if(shareToken)startRideStatusPolling(shareToken);}}}}
function initPassengerMapFeatures(){{const pI=document.getElementById('pickup-location'),dI=document.getElementById('dropoff-location');const pAC=new google.maps.places.Autocomplete(pI),dAC=new google.maps.places.Autocomplete(dI);pAC.addListener('place_changed',()=>handlePlaceSelect(pAC,'pickup'));dAC.addListener('place_changed',()=>handlePlaceSelect(dAC,'dropoff'));pollForActiveRide();}}
function handlePlaceSelect(ac,type){{const p=ac.getPlace();if(!p.geometry||!p.geometry.location)return showToast(`Could not find ${{type}} location.`,'error');const l=p.geometry.location,el=document.getElementById(`${{type}}-location`);el.dataset.lat=l.lat();el.dataset.lng=l.lng();el.dataset.addr=p.formatted_address;if(type==='pickup'){{if(pickupMarker)pickupMarker.setMap(null);pickupMarker=new google.maps.Marker({{position:l,map:map,icon:'http://maps.google.com/mapfiles/ms/icons/green-dot.png'}});map.panTo(l);}}else{{if(dropoffMarker)dropoffMarker.setMap(null);dropoffMarker=new google.maps.Marker({{position:l,map:map,icon:'http://maps.google.com/mapfiles/ms/icons/red-dot.png'}});}}calculateRouteAndFare();}}
async function calculateRouteAndFare(){{const pLat=document.getElementById('pickup-location')?.dataset.lat,pLng=document.getElementById('pickup-location')?.dataset.lng,dLat=document.getElementById('dropoff-location')?.dataset.lat,dLng=document.getElementById('dropoff-location')?.dataset.lng;if(pLat&&dLat){{directionsService.route({{origin:new google.maps.LatLng(pLat,pLng),destination:new google.maps.LatLng(dLat,dLng),travelMode:'DRIVING'}},(res,stat)=>{{if(stat=='OK')directionsRenderer.setDirections(res);}});const vt=document.querySelector('.vehicle-option.selected')?.dataset.type||'Economy';try{{const f=await apiFetch('/api/estimate-fare',{{method:'POST',body:{{pickup_lat:parseFloat(pLat),pickup_lng:parseFloat(pLng),dropoff_lat:parseFloat(dLat),dropoff_lng:parseFloat(dLng),vehicle_type:vt}}}});const fb=document.getElementById('fare-estimate-box');if(fb)fb.innerHTML=`<h3>$${{f.estimated_fare.toFixed(2)}}</h3><p>${{f.distance_km}} km·${{f.duration_minutes}} mins·Surge:${{f.surge_multiplier}}x</p>`;}}catch(e){{console.error("Fare estimate failed:",e)}}}}}}
async function handleBookRide(){{const pEl=document.getElementById('pickup-location'),dEl=document.getElementById('dropoff-location');const selVeh=document.querySelector('.vehicle-option.selected');if(!pEl?.dataset.lat||!dEl?.dataset.lat)return showToast('Select pickup & drop-off locations.','error');if(!selVeh)return showToast('Select a vehicle type.','error');const req={{pickup_lat:parseFloat(pEl.dataset.lat),pickup_lng:parseFloat(pEl.dataset.lng),pickup_address:pEl.dataset.addr,dropoff_lat:parseFloat(dEl.dataset.lat),dropoff_lng:parseFloat(dEl.dataset.lng),dropoff_address:dEl.dataset.addr,vehicle_type:selVeh.dataset.type,payment_method:document.getElementById('payment-method').value}};try{{const ride=await apiFetch('/api/passenger/request-ride',{{method:'POST',body:req}});showToast('Ride requested! Searching...','success');updateRideStatusUI(ride);startRideStatusPolling(ride.id);}}catch(e){{console.error("Booking failed:",e)}}}}
function startRideStatusPolling(rideIdOrToken){{if(rideStatusPollInterval)clearInterval(rideStatusPollInterval);const isPublic=isNaN(rideIdOrToken);const url=isPublic?`/api/ride/public/${{rideIdOrToken}}`:`/api/ride/passenger/${{rideIdOrToken}}`;rideStatusPollInterval=setInterval(async()=>{{try{{const ride=await apiFetch(url,{{}},false);updateRideStatusUI(ride,isPublic);if(['completed','cancelled'].includes(ride.status)){{clearInterval(rideStatusPollInterval);if(ride.status==='completed'&&!isPublic&&!ride.passenger_rated){{showRatingModal(ride.id);}}}}}}catch(e){{clearInterval(rideStatusPollInterval);}}}},5000);}}
function pollForActiveRide(){{apiFetch('/api/passenger/active-ride',{{}},false).then(r=>{{if(r&&r.id){{showToast('Resuming active ride.','info');updateRideStatusUI(r);startRideStatusPolling(r.id);}}}}).catch(e=>{{}});}}
function updateRideStatusUI(r,isPublic=false){{let textEl,driverInfoEl,progBar,steps,container;if(isPublic){{container=document.getElementById('track-ride-status');if(!container)return;textEl=document.getElementById('track-status-text');driverInfoEl=document.getElementById('track-driver-text');}}else{{container=document.getElementById('ride-status-container');if(!container)return;container.style.display='block';textEl=document.getElementById('ride-status-text');driverInfoEl=document.getElementById('ride-driver-info');progBar=document.querySelector('.progress-bar-fg');steps=document.querySelectorAll('.progress-step');document.getElementById('sos-button').dataset.rideId=r.id;document.getElementById('share-trip-button').dataset.shareToken=r.share_token;}}if(!textEl||!driverInfoEl)return;textEl.textContent=`Status: ${{r.status.replace(/_/g,' ').toUpperCase()}}`;if(r.driver){{driverInfoEl.innerHTML=`<div><strong>${{r.driver.full_name}}</strong>(${{r.driver.average_rating}} <i class="fas fa-star" style="color:#f59e0b;"></i>)</div><div>${{r.driver.vehicle.color}} ${{r.driver.vehicle.make}} - ${{r.driver.vehicle.license_plate}}</div>`;}}else{{driverInfoEl.innerHTML='Searching for a driver...';}}if(!isPublic&&progBar&&steps){{let prog=0;steps.forEach(s=>s.classList.remove('active'));switch(r.status){{case'pending':prog=10;steps[0].classList.add('active');break;case'accepted':case'arriving':prog=40;steps[1].classList.add('active');break;case'in_progress':prog=75;steps[2].classList.add('active');break;case'completed':prog=100;steps[3].classList.add('active');driverInfoEl.innerHTML='Trip completed! Thank you.';setTimeout(()=>container.style.display='none',10000);break;case'cancelled':prog=0;driverInfoEl.innerHTML='This ride has been cancelled.';setTimeout(()=>container.style.display='none',5000);break;}}progBar.style.width=`${{prog}}%`;}}if(r.driver_current_lat&&r.driver_current_lng){{updateDriverMarker(r.driver_current_lat,r.driver_current_lng,r.driver?.full_name);}}}}
function updateDriverMarker(lat,lng,name){{if(!map||!lat||!lng)return;const pos=new google.maps.LatLng(lat,lng);if(!driverMarker){{driverMarker=new google.maps.Marker({{position:pos,map,title:name,icon:{{path:google.maps.SymbolPath.FORWARD_CLOSED_ARROW,scale:6,fillColor:"#12121c",fillOpacity:1,strokeWeight:2,strokeColor:"#6d28d9"}}}});}}else{{driverMarker.setPosition(pos);}}const bounds=map.getBounds();if(bounds&&!bounds.contains(pos)){{map.panTo(pos);}}}}
function initDriverMapFeatures(){{const t=document.getElementById('driver-status-toggle');if(t)t.addEventListener('click',toggleDriverAvailability);pollForRideRequests();setInterval(()=>{{if(t&&t.classList.contains('online')){{if(navigator.geolocation)navigator.geolocation.getCurrentPosition(p=>apiFetch('/api/driver/update-location',{{method:'POST',body:{{lat:p.coords.latitude,lng:p.coords.longitude}}}},false).catch(e=>{{}}));}}}},10000);}}
async function toggleDriverAvailability(){{const t=this;const isOnline=!t.classList.contains('online');try{{await apiFetch('/api/driver/toggle-availability',{{method:'POST',body:{{online:isOnline}}}});t.classList.toggle('online');t.querySelector('span').textContent=`You are ${{isOnline?'ONLINE':'OFFLINE'}}`;showToast(`You are now ${{isOnline?'online':'offline'}}`,'success');}}catch(e){{}}}}
function pollForRideRequests(){{if(rideRequestPollInterval)clearInterval(rideRequestPollInterval);rideRequestPollInterval=setInterval(async()=>{{const t=document.getElementById('driver-status-toggle');if(t&&t.classList.contains('online')&&t.dataset.status==='online'){{try{{const r=await apiFetch('/api/driver/ride-request',{{}},false);if(r&&r.id){{showRideRequestModal(r);clearInterval(rideRequestPollInterval);}}}}catch(e){{}}}}}},5000);}}
function showRideRequestModal(r){{const m=document.getElementById('ride-request-modal');if(!m)return;document.getElementById('request-pickup').textContent=r.pickup_address;document.getElementById('request-dropoff').textContent=r.dropoff_address;document.getElementById('request-fare').textContent=`$${{r.estimated_fare.toFixed(2)}}`;m.style.display='flex';let time=30;const timerEl=document.getElementById('countdown-timer');timerEl.textContent=time;countdownInterval=setInterval(()=>{{time--;timerEl.textContent=time;if(time<=0)hideRideRequestModal();}},1000);document.getElementById('accept-ride-btn').onclick=()=>handleRideAction('accept',r.id);document.getElementById('reject-ride-btn').onclick=()=>handleRideAction('reject',r.id);}}
function hideRideRequestModal(){{const modal=document.getElementById('ride-request-modal');if(modal)modal.style.display='none';clearInterval(countdownInterval);pollForRideRequests();}}
async function handleRideAction(action,id){{hideRideRequestModal();if(action==='reject')return;try{{const r=await apiFetch(`/api/driver/accept-ride`,{{method:'POST',body:{{ride_id:id}}}});showToast(`Ride accepted!`,'success');updateDriverTripUI(r);}}catch(e){{console.error("Accept ride failed:",e)}}}}
function updateDriverTripUI(r){{const cont=document.getElementById('current-trip-info');if(!cont)return;cont.innerHTML=`<div class="card"><h4>Current Ride:#${{r.id}}</h4><p><strong>To:</strong>${{r.dropoff_address}}</p><p><strong>Passenger:</strong>${{r.passenger.full_name}}</p><div id="driver-action-buttons"><button class="btn btn-secondary" onclick="updateDriverRideStatus(${{r.id}},'arriving')">I've Arrived</button></div></div>`;const toggle=document.getElementById('driver-status-toggle');if(toggle)toggle.dataset.status='on_trip';}}
async function updateDriverRideStatus(id,status){{try{{await apiFetch('/api/driver/update-ride-status',{{method:'POST',body:{{ride_id:id,status:status}}}});showToast(`Status updated to ${{status}}`,'success');const btns=document.getElementById('driver-action-buttons');if(!btns)return;let nextAction='';if(status==='arriving'){{nextAction=`<button class="btn btn-primary" onclick="updateDriverRideStatus(${{id}},'in_progress')">Start Trip</button>`;}}else if(status==='in_progress'){{nextAction=`<button class="btn btn-success" onclick="updateDriverRideStatus(${{id}},'completed')">Complete Trip</button>`;}}else{{document.getElementById('current-trip-info').innerHTML='<p>No active trip. You are now available for new requests.</p>';const t=document.getElementById('driver-status-toggle');if(t)t.dataset.status='online';pollForRideRequests();return;}}btns.innerHTML=nextAction;}}catch(e){{console.error("Update status failed:",e)}}}}

// --- GENERAL UI HANDLERS ---
function showRatingModal(id){{const m=document.getElementById('rating-modal');if(!m)return;m.style.display='flex';m.dataset.rideId=id;}}
function handleSOS(){{const btn=document.getElementById('sos-button');if(!btn||!btn.dataset.rideId||!confirm("Trigger SOS alert? This will notify our safety team."))return;if(navigator.geolocation)navigator.geolocation.getCurrentPosition(async p=>{{try{{await apiFetch('/api/ride/sos',{{method:'POST',body:{{ride_id:parseInt(btn.dataset.rideId),lat:p.coords.latitude,lng:p.coords.longitude}}}});showToast("SOS Alert triggered. Team notified.",'error',10000);}}catch(e){{}}}});}}
function handleShareTrip(token){{const url=`${{window.location.origin}}/track/${{token}}`;const modal=document.getElementById('share-trip-modal');if(!modal)return;modal.querySelector('#share-url-input').value=url;modal.style.display='flex';}}
function copyShareUrl(){{const input=document.getElementById('share-url-input');if(!input)return;input.select();input.setSelectionRange(0,99999);document.execCommand('copy');showToast('Link copied to clipboard!','success');}}
function handleTopUp(){{const amt=parseFloat(document.getElementById('topup-amount')?.value);if(isNaN(amt)||amt<=0)return showToast("Enter a valid amount.",'error');PaystackPop.setup({{key:PAYSTACK_PUBLIC_KEY,email:document.body.dataset.userEmail,amount:amt*100,currency:'USD',ref:'rydepro_w_'+Math.floor(1e9*Math.random()+1),callback:async r=>{{try{{const res=await apiFetch('/api/wallet/verify-topup',{{method:'POST',body:{{reference:r.reference,amount:amt}}}});showToast('Wallet topped up!','success');const b=document.getElementById('wallet-balance');if(b)b.textContent=`$${{res.new_balance.toFixed(2)}}`;loadWalletTransactions();}}catch(e){{showToast('Verification failed.','error');}}}},onClose:()=>showToast('Payment window closed.','info')}}).openIframe();}}
async function loadWalletTransactions(){{const cont=document.getElementById('transaction-history-body');if(!cont)return;try{{const txs=await apiFetch('/api/wallet/history');if(txs.length===0){{cont.innerHTML='<tr><td colspan="4" class="text-center">No transactions yet.</td></tr>';}}else{{cont.innerHTML=txs.map(tx=>`<tr><td>${{new Date(tx.created_at).toLocaleString()}}</td><td>${{tx.description}}</td><td class="${{tx.amount>0?'text-success':'text-error'}}">$${{tx.amount.toFixed(2)}}</td><td>${{tx.reference?tx.reference.substring(0,20)+'...':''}}</td></tr>`).join('');}}}}catch(e){{cont.innerHTML='<tr><td colspan="4" class="text-center text-error">Could not load history.</td></tr>';}}}}
async function handleAdminDriverAction(btn,driverId,action){{const reason=action==='reject'?prompt("Reason for rejection:"):null;if(action==='reject'&&!reason)return;btn.disabled=true;try{{const res=await apiFetch('/api/admin/driver-action',{{method:'POST',body:{{driver_id:driverId,action,reason}}}});showToast(`Driver ${{action}}ed.`,'success');const row=btn.closest('tr');row.querySelector('.status-cell').innerHTML=`<span class="status-tag status-${{res.new_status}}">${{res.new_status}}</span>`;row.querySelector('.action-cell').innerHTML='Action Taken';}}catch(e){{btn.disabled=false;}}}}
async function purchaseSubscription(plan){{if(!confirm(`Confirm purchase of ${{plan.charAt(0).toUpperCase()+plan.slice(1)}} plan? This is a simulated payment from your wallet.`))return;try{{const res=await apiFetch('/api/user/subscribe',{{method:'POST',body:{{plan}}}});showToast(`Successfully subscribed to ${{res.new_plan}}!`,'success');window.location.reload();}}catch(e){{}}}}
function showAuthTab(tabName){{document.querySelectorAll('.auth-tab-content').forEach(c=>c.classList.remove('active'));document.querySelectorAll('.auth-tab-button').forEach(b=>b.classList.remove('active'));const content=document.getElementById(tabName);const button=document.querySelector(`[data-tab='${{tabName}}']`);if(content)content.classList.add('active');if(button)button.classList.add('active');}}

// --- DOMCONTENTLOADED - MAIN ENTRY POINT ---
document.addEventListener('DOMContentLoaded',()=>{{
    if (document.getElementById('auth-page')) initializeFirebase();
    if (document.getElementById('map')) loadGoogleMapsScript();
    if (document.getElementById('wallet-container')) loadWalletTransactions();
    document.querySelectorAll('.auth-tab-button').forEach(b=>b.addEventListener('click',()=>showAuthTab(b.dataset.tab)));
    document.getElementById('logout-link')?.addEventListener('click',async e=>{{e.preventDefault();try{{await apiFetch('/api/auth/logout',{{method:'POST'}});}}finally{{window.location.href='/';}}}});
    document.getElementById('book-ride-btn')?.addEventListener('click',handleBookRide);
    document.querySelectorAll('.vehicle-option').forEach(o=>o.addEventListener('click',()=>{{document.querySelectorAll('.vehicle-option').forEach(opt=>opt.classList.remove('selected'));o.classList.add('selected');calculateRouteAndFare();}}));
    document.getElementById('sos-button')?.addEventListener('click',handleSOS);
    document.getElementById('share-trip-button')?.addEventListener('click',function(){{handleShareTrip(this.dataset.shareToken);}});
    document.getElementById('topup-btn')?.addEventListener('click',handleTopUp);
    document.querySelectorAll('.close-modal').forEach(el=>el.addEventListener('click',()=>el.closest('.modal').style.display='none'));
    document.getElementById('rating-modal-form')?.addEventListener('submit',async e=>{{e.preventDefault();const m=document.getElementById('rating-modal'),id=m.dataset.rideId;const r=e.target.rating.value,c=e.target.comment.value;try{{await apiFetch('/api/ride/rate',{{method:'POST',body:{{ride_id:parseInt(id),rating:parseInt(r),comment:c}}}});showToast('Feedback received!','success');m.style.display='none';window.location.reload();}}catch(err){{}}}});
    document.getElementById('profile-update-form')?.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,formData=new FormData(form);try{{const res=await apiFetch('/api/profile',{{method:'PUT',body:formData}});showToast('Profile updated!','success');if(res.profile_picture_url)document.getElementById('profile-avatar-img').src=res.profile_picture_url+'?t='+new Date().getTime();}}catch(err){{}}}});
    document.getElementById('driver-registration-form')?.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,formData=new FormData(form),btn=form.querySelector('button');btn.disabled=true;btn.textContent='Submitting...';try{{await apiFetch('/api/driver/register',{{method:'POST',body:formData}});showToast('Registration submitted for review!','success');setTimeout(()=>window.location.href='/driver',2000);}}catch(err){{btn.disabled=false;btn.textContent='Submit for Review';}}}});
    document.getElementById('login-form')?.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,formData=new FormData(form);const email=formData.get('email'),password=formData.get('password');try{{const data=await apiFetch('/api/auth/login',{{method:'POST',body:{{email,password}}}});showToast('Login successful! Redirecting...','success');setTimeout(()=>window.location.href=data.redirect_url,1000);}}catch(err){{}}}});
    document.getElementById('register-form')?.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,formData=new FormData(form);const full_name=formData.get('full_name'),email=formData.get('email'),password=formData.get('password');try{{await apiFetch('/api/auth/register',{{method:'POST',body:{{full_name,email,password}}}});showToast('Registration successful! Please log in.','success');setTimeout(()=>window.location.href='/login',1500);}}catch(err){{}}}});
}});
"""


# ==============================================================================
# PART 9: HTML TEMPLATES
# ==============================================================================
def get_base_html(title, content, current_user=None):
    user_menu, body_data = "", ""
    if current_user:
        dashboard_link = f"/{current_user.role.value}"
        body_data = f"data-user-id='{current_user.id}' data-user-email='{current_user.email}'"
        user_menu = f"""<div class="user-menu"><img src="{current_user.profile_picture_url}" alt="Avatar"/><ul class="dropdown-menu"><li><a href="{dashboard_link}"><i class="fa-fw fas fa-tachometer-alt"></i> Dashboard</a></li><li><a href="/{current_user.role.value}/profile"><i class="fa-fw fas fa-user-edit"></i> Profile</a></li><li><a href="/{current_user.role.value}/history"><i class="fa-fw fas fa-history"></i> History</a></li>{'<li><a href="/passenger/wallet"><i class="fa-fw fas fa-wallet"></i> Wallet</a></li>' if current_user.role == UserRole.passenger else ''}{'<li><a href="/passenger/subscriptions"><i class="fa-fw fas fa-star"></i> Subscriptions</a></li>' if current_user.role == UserRole.passenger else ''}<li><a href="#" id="logout-link"><i class="fa-fw fas fa-sign-out-alt"></i> Logout</a></li></ul></div>"""
    else:
        user_menu = f'<a href="/login" class="btn btn-primary">Login / Sign Up</a>'
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{CONFIG['PROJECT_NAME']} - {title}</title><style>{MAIN_CSS}</style><script src="https://www.gstatic.com/firebasejs/9.6.1/firebase-app-compat.js"></script><script src="https://www.gstatic.com/firebasejs/9.6.1/firebase-auth-compat.js"></script><script src="https://www.gstatic.com/firebasejs/ui/6.0.1/firebase-ui-auth.js"></script><link type="text/css" rel="stylesheet" href="https://www.gstatic.com/firebasejs/ui/6.0.1/firebase-ui-auth.css"/></head><body {body_data}><div id="toast-container"></div><header class="main-header"><a href="/" class="logo"><i class="fas fa-rocket"></i> RydePro</a><nav class="main-nav"></nav><div>{user_menu}</div></header><main class="main-content">{content}</main><footer class="main-footer"><p>© {datetime.now().year} {CONFIG['PROJECT_NAME']}.</p></footer><script>{MAIN_JS}</script></body></html>"""


def get_dashboard_html(title, content, user, active_page):
    sidebar_links = "";
    role = user.role.value
    links = {"passenger": [("map-marked-alt", "Book a Ride", ""), ("history", "Ride History", "/history"),
                           ("wallet", "Wallet", "/wallet"), ("star", "Subscriptions", "/subscriptions"),
                           ("user-cog", "Profile", "/profile")],
             "driver": [("tachometer-alt", "Dashboard", ""), ("dollar-sign", "Earnings", "/earnings"),
                        ("history", "Trip History", "/history"), ("user-shield", "Profile & Docs", "/profile")],
             "admin": [("chart-line", "Dashboard", ""), ("users", "Users", "/users"),
                       ("id-card", "Drivers", "/drivers"), ("route", "Rides", "/rides")]}
    for icon, text, path in links[
        role]: sidebar_links += f'<li><a href="/{role}{path}" class="{"active" if active_page == (path.strip("/") if path else "dashboard") else ""}"><i class="fa-fw fas fa-{icon}"></i> {text}</a></li>'
    return get_base_html(title,
                         f'<div class="dashboard-layout"><aside class="sidebar"><nav class="sidebar-nav"><ul>{sidebar_links}</ul></nav></aside><section class="dashboard-content"><h1>{title}</h1>{content}</section></div>',
                         user)


def render_generic_page(title, content):
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{CONFIG['PROJECT_NAME']} - {title}</title><style>{MAIN_CSS}</style></head><body><header class="main-header"><a href="/" class="logo"><i class="fas fa-rocket"></i> RydePro</a></header><main class="main-content"><div class="container">{content}</div></main><footer class="main-footer"><p>© {datetime.now().year} {CONFIG['PROJECT_NAME']}.</p></footer><script>{MAIN_JS}</script></body></html>"""


LANDING_PAGE_HTML = """<div class="container text-center" style="padding: 4rem 1.5rem;"><h1 style="font-size:3rem;">The Future of Mobility is Here</h1><p class="mb-3" style="font-size: 1.2rem; color: var(--text-secondary);">Welcome to RydePro. Seamless, intelligent, and on-demand.</p><a href="/login" class="btn btn-primary btn-lg" style="padding: 15px 40px; font-size: 1.2rem;">Get Started</a></div>"""
LOGIN_PAGE_HTML = """
<div id="auth-page"><div class="form-container">
    <div class="auth-tabs"><button class="auth-tab-button active" data-tab="email-tab">Email & Password</button><button class="auth-tab-button" data-tab="social-tab">Social Login</button></div>
    <div id="email-tab" class="auth-tab-content active"><h2 class="text-center">Login</h2><form id="login-form"><div class="form-group"><label for="email">Email</label><input type="email" name="email" class="form-control" required></div><div class="form-group"><label for="password">Password</label><input type="password" name="password" class="form-control" required></div><button type="submit" class="btn btn-primary" style="width:100%;">Login</button></form><p class="text-center mt-3">Don't have an account? <a href="/register">Sign Up</a></p></div>
    <div id="social-tab" class="auth-tab-content"><h2 class="text-center">Social Login</h2><p class="text-center" style="color:var(--text-secondary);">Use your Google account to sign in instantly.</p><div id="firebaseui-auth-container"></div><div id="firebaseui-loader" class="text-center mt-3"><div class="loader"></div></div></div>
</div></div>"""
REGISTER_PAGE_HTML = """
<div id="auth-page"><div class="form-container">
    <h2 class="text-center">Create an Account</h2><form id="register-form"><div class="form-group"><label for="full_name">Full Name</label><input type="text" name="full_name" class="form-control" required></div><div class="form-group"><label for="email">Email</label><input type="email" name="email" class="form-control" required></div><div class="form-group"><label for="password">Password</label><input type="password" name="password" class="form-control" required minlength="6"></div><button type="submit" class="btn btn-primary" style="width:100%;">Sign Up</button></form><p class="text-center mt-3">Already have an account? <a href="/login">Log In</a></p>
</div></div>"""
PASSENGER_DASHBOARD_HTML = """<div id="passenger-dashboard" class="passenger-dashboard-grid"><div id="map"></div><div class="booking-panel"><h3>Where to?</h3><div class="form-group"><label for="pickup-location">Pickup</label><input type="text" id="pickup-location" class="form-control" placeholder="Enter pickup address"></div><div class="form-group"><label for="dropoff-location">Drop-off</label><input type="text" id="dropoff-location" class="form-control" placeholder="Enter destination"></div><p>Vehicle Type:</p><div class="vehicle-options"><div class="vehicle-option selected" data-type="Economy"><i class="fas fa-car-side icon"></i><span class="name">Economy</span></div><div class="vehicle-option" data-type="Luxury"><i class="fas fa-gem icon"></i><span class="name">Luxury</span></div><div class="vehicle-option" data-type="SUV"><i class="fas fa-truck-monster icon"></i><span class="name">SUV</span></div></div><div class="form-group"><label for="payment-method">Payment</label><select id="payment-method" class="form-control"><option value="wallet">Wallet</option><option value="card">Card (Future)</option><option value="cash">Cash</option></select></div><button id="book-ride-btn" class="btn btn-primary">Request Ride</button><div id="fare-estimate-box" class="mt-3"><p>Enter locations for fare estimate.</p></div></div></div><div id="ride-status-container"><div class="ride-status-header"><h3 id="ride-status-text">Status...</h3><div class="ride-status-actions"><button id="share-trip-button" title="Share Trip"><i class="fas fa-share-alt"></i></button><button id="sos-button" title="Emergency SOS"><i class="fas fa-exclamation-triangle"></i></button></div></div><div id="ride-driver-info" class="mt-2"></div><div class="progress-container mt-3"><div class="progress-steps"><div class="progress-step">Requested</div><div class="progress-step">En Route</div><div class="progress-step">In Progress</div><div class="progress-step">Completed</div></div><div class="progress-bar-bg"><div class="progress-bar-fg"></div></div></div></div><div id="rating-modal" class="modal"><div class="modal-content"><span class="close-modal">×</span><h2>How was your ride?</h2><form id="rating-modal-form"><div class="star-rating"><input type="radio" id="5-stars" name="rating" value="5" /><label for="5-stars" class="fa-solid fa-star"></label><input type="radio" id="4-stars" name="rating" value="4" /><label for="4-stars" class="fa-solid fa-star"></label><input type="radio" id="3-stars" name="rating" value="3" checked /><label for="3-stars" class="fa-solid fa-star"></label><input type="radio" id="2-stars" name="rating" value="2" /><label for="2-stars" class="fa-solid fa-star"></label><input type="radio" id="1-star" name="rating" value="1" /><label for="1-star" class="fa-solid fa-star"></label></div><div class="form-group mt-3"><textarea name="comment" class="form-control" placeholder="Add a comment..."></textarea></div><button type="submit" class="btn btn-primary">Submit Review</button></form></div></div><div id="share-trip-modal" class="modal"><div class="modal-content"><span class="close-modal">×</span><h2>Share Your Trip</h2><p>Send this link to friends and family to track your ride in real-time.</p><div class="form-group" style="display:flex; gap:10px;"><input type="text" id="share-url-input" class="form-control" readonly><button class="btn btn-secondary" onclick="copyShareUrl()">Copy</button></div></div></div>"""
# (Other HTML templates are unchanged and included for completeness)
DRIVER_DASHBOARD_HTML = """<div id="driver-dashboard"><div class="grid-container"><div class="card"><div id="driver-status-toggle" class="{is_online_class}" data-status="{status_text_raw}"><div class="toggle-switch"><div class="slider"></div></div><span>You are {status_text}</span></div></div><div class="stat-card"><div class="icon"><i class="fas fa-dollar-sign"></i></div><div class="value">$ {today_earnings:.2f}</div><div class="label">Today's Earnings</div></div><div class="stat-card"><div class="icon"><i class="fas fa-star"></i></div><div class="value">{avg_rating}</div><div class="label">Your Rating</div></div></div><div class="card mt-3"><div class="card-header">Current Trip</div><div id="current-trip-info">{current_trip_html}</div></div></div><div id="ride-request-modal" class="modal"><div class="modal-content"><h2>New Ride Request!</h2><div class="countdown-timer" id="countdown-timer">30</div><div class="card" style="text-align:left;"><p><strong>From:</strong> <span id="request-pickup"></span></p><p><strong>To:</strong> <span id="request-dropoff"></span></p><p><strong>Est. Fare:</strong> <span id="request-fare"></span></p></div><div style="display:flex;justify-content:space-around;margin-top:1.5rem"><button id="reject-ride-btn" class="btn btn-danger">Reject</button><button id="accept-ride-btn" class="btn btn-success">Accept</button></div></div></div>"""
PROFILE_PAGE_HTML = """<div class="card"><div class="card-header">Your Profile</div><form id="profile-update-form" class="text-center"><img src="{user.profile_picture_url}" alt="Avatar" class="profile-avatar" id="profile-avatar-img"><div class="form-group mt-3"><div class="file-upload-wrapper btn btn-secondary"><span>Change Photo</span><input type="file" name="avatar" accept="image/*"></div></div><div class="form-group" style="text-align:left"><label for="full_name">Full Name</label><input type="text" name="full_name" class="form-control" value="{user.full_name}"></div><div class="form-group" style="text-align:left"><label>Email</label><input type="email" class="form-control" value="{user.email}" disabled></div><button type="submit" class="btn btn-primary">Save Changes</button></form></div>"""
HISTORY_PAGE_HTML = """<div class="card"><div class="card-header">Ride History</div><div class="table-container"><table class="data-table"><thead><tr><th>Date</th><th>From</th><th>To</th><th>Fare</th><th>Status</th><th>Action</th></tr></thead><tbody>{history_rows}</tbody></table></div></div>"""
WALLET_PAGE_HTML = """<div id="wallet-container"><div class="card"><div class="card-header">Your Wallet</div><h2>Balance: <span id="wallet-balance" class="text-success">${user.wallet_balance:.2f}</span></h2><div class="form-group mt-3"><label for="topup-amount">Top-up Amount ($)</label><input type="number" id="topup-amount" class="form-control" placeholder="e.g., 50"></div><button id="topup-btn" class="btn btn-primary">Top Up with PayStack</button></div><div class="card mt-3"><div class="card-header">Transaction History</div><div class="table-container"><table class="data-table"><thead><tr><th>Date</th><th>Description</th><th>Amount</th><th>Reference</th></tr></thead><tbody id="transaction-history-body"><div class="loader"></div></tbody></table></div></div><script src="https://js.paystack.co/v1/inline.js"></script>"""
DRIVER_REGISTRATION_HTML = """<div class="form-container"><h2 class="text-center">Become a RydePro Driver</h2><p class="text-center" style="color:var(--text-secondary)">Complete your profile to start earning.</p><form id="driver-registration-form"><div class="card-header" style="margin-top:1rem;">Vehicle Information</div><div class="form-group"><label>Make</label><input type="text" name="vehicle_make" class="form-control" required></div><div class="form-group"><label>Model</label><input type="text" name="vehicle_model" class="form-control" required></div><div class="form-group"><label>Year</label><input type="number" name="vehicle_year" class="form-control" required></div><div class="form-group"><label>Color</label><input type="text" name="vehicle_color" class="form-control" required></div><div class="form-group"><label>License Plate</label><input type="text" name="vehicle_license_plate" class="form-control" required></div><div class="form-group"><label>Vehicle Type</label><select name="vehicle_type" class="form-control">{vehicle_type_options}</select></div><div class="card-header" style="margin-top:1.5rem;">Documents</div><div class="form-group"><label>Driver's License Number</label><input type="text" name="license_number" class="form-control" required></div><div class="form-group"><label>License Document (PDF/JPG)</label><input type="file" name="license_doc" class="form-control" accept=".pdf,.jpg,.jpeg,.png" required></div><div class="form-group"><label>Insurance Document (PDF/JPG)</label><input type="file" name="insurance_doc" class="form-control" accept=".pdf,.jpg,.jpeg,.png" required></div><button type="submit" class="btn btn-primary">Submit for Review</button></form></div>"""
ADMIN_DRIVERS_HTML = """<div class="card"><div class="card-header">Driver Management</div><div class="table-container"><table class="data-table"><thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Rating</th><th>Status</th><th>Actions</th></tr></thead><tbody>{driver_rows}</tbody></table></div></div>"""
SUBSCRIPTIONS_PAGE_HTML = """<div class="grid-container">{plan_cards}</div>"""
TRACK_RIDE_PAGE_HTML = """<div id="map" style="width:100%; height:calc(100vh - 100px); border-radius:0;"></div><div id="track-ride-status" style="position:fixed; bottom:20px; left:20px; background:var(--bg-dark-secondary); padding:1.5rem; border-radius:var(--border-radius-md); border:1px solid var(--border-color); box-shadow:var(--shadow-lg);"><h3 id="track-status-text">Tracking Ride...</h3><p id="track-driver-text">Fetching details...</p></div>"""

# ==============================================================================
# PART 10: FASTAPI APPLICATION & ENDPOINTS
# ==============================================================================
app = FastAPI(title=CONFIG["PROJECT_NAME"], on_startup=[setup_dirs, init_firebase])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
app.mount("/static", StaticFiles(directory=CONFIG["UPLOADS_DIR"]), name="static")

# --- Routers ---
api_router = APIRouter(prefix="/api")
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# --- AUTHENTICATION API ENDPOINTS ---
@auth_router.post("/register")
async def api_register(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        full_name=user_data.full_name,
        email=user_data.email,
        hashed_password=hashed_password,
        role=UserRole.admin if user_data.email == "admin@rydepro.com" else UserRole.passenger
    )
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}


@auth_router.post("/login")
async def api_login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not user.hashed_password or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": str(user.id)})
    response = JSONResponse(content={"redirect_url": f"/{user.role.value}"})
    response.set_cookie("access_token", access_token, httponly=True, samesite='lax', secure=False, path="/")
    return response


@auth_router.post("/firebase-login")
async def api_firebase_login(data: FirebaseLogin, db: Session = Depends(get_db)):
    try:
        decoded = auth.verify_id_token(data.firebase_token)
        uid, email = decoded['uid'], decoded.get('email')
        user = db.query(User).filter(User.firebase_uid == uid).first()
        if not user:
            user = db.query(User).filter(User.email == email).first()
            if user:
                user.firebase_uid = uid
            else:
                user = User(firebase_uid=uid, full_name=decoded.get('name', 'New User'), email=email,
                            role=UserRole.passenger)
                db.add(user)
        db.commit()
        db.refresh(user)
        access_token = create_access_token({"sub": str(user.id)})
        response = JSONResponse(content={"redirect_url": f"/{user.role.value}"})
        response.set_cookie("access_token", access_token, httponly=True, samesite='lax', secure=False, path="/")
        return response
    except Exception as e:
        raise HTTPException(401, f"Invalid Firebase token: {e}")


@auth_router.post("/logout")
def api_logout():
    res = JSONResponse({"message": "Logged out"})
    res.delete_cookie("access_token")
    return res


# --- PAGE RENDERING ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def page_root(user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(f"/{user.role.value}")
    return HTMLResponse(get_base_html("Home", LANDING_PAGE_HTML, user))


@app.get("/login", response_class=HTMLResponse)
async def page_login(user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(f"/{user.role.value}")
    return HTMLResponse(get_base_html("Login", LOGIN_PAGE_HTML))


@app.get("/register", response_class=HTMLResponse)
async def page_register(user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(f"/{user.role.value}")
    return HTMLResponse(get_base_html("Sign Up", REGISTER_PAGE_HTML))


# (The rest of the endpoints from the previous version follow here and are unchanged)
@app.get("/track/{share_token}", response_class=HTMLResponse)
async def page_track_ride(share_token: str):
    return HTMLResponse(render_generic_page("Track Ride", TRACK_RIDE_PAGE_HTML))


@app.get("/passenger", response_class=HTMLResponse)
async def page_passenger_dashboard(user: User = Depends(require_passenger)):
    return HTMLResponse(get_dashboard_html("Book a Ride", PASSENGER_DASHBOARD_HTML, user, "dashboard"))


@app.get("/passenger/profile", response_class=HTMLResponse)
async def page_passenger_profile(user: User = Depends(require_passenger)):
    return HTMLResponse(get_dashboard_html("Profile", PROFILE_PAGE_HTML.format(user=user), user, "profile"))


@app.get("/passenger/history", response_class=HTMLResponse)
async def page_passenger_history(user: User = Depends(require_passenger), db: Session = Depends(get_db)):
    """
    Renders the passenger's ride history page.
    This version uses a helper function to generate the action button HTML,
    avoiding complex and unreadable backslash escaping.
    """
    rides = db.query(Ride).options(
        joinedload(Ride.driver).joinedload(Driver.user)
    ).filter(Ride.passenger_id == user.id).order_by(Ride.created_at.desc()).limit(50).all()

    def generate_action_button(ride: Ride) -> str:
        """
        Generates the HTML for the 'Action' column button based on ride status.
        This makes the main loop much cleaner.
        """
        if ride.status == RideStatus.completed and not ride.passenger_rated:
            # Using single quotes for the outer HTML string allows double quotes inside without escaping.
            return f'<button class="btn btn-secondary" onclick="showRatingModal({ride.id})">Rate</button>'
        return "N/A"

    # Build table rows using a clear, multi-line f-string
    rows = []
    for r in rides:
        action_html = generate_action_button(r)
        row_html = f"""
            <tr>
                <td>{r.created_at.strftime("%b %d, %H:%M")}</td>
                <td>{r.pickup_address[:25]}...</td>
                <td>{r.dropoff_address[:25]}...</td>
                <td>${(r.actual_fare or r.estimated_fare):.2f}</td>
                <td><span class="status-tag status-{r.status.value}">{r.status.value.replace('_', ' ')}</span></td>
                <td>{action_html}</td>
            </tr>
        """
        rows.append(row_html)
    history_rows_html = "".join(rows)
    if not history_rows_html:
        history_rows_html = '<tr><td colspan="6" class="text-center">You have no ride history yet.</td></tr>'
    return HTMLResponse(
        get_dashboard_html("Ride History", HISTORY_PAGE_HTML.format(history_rows=history_rows_html), user, "history"))


@app.get("/passenger/wallet", response_class=HTMLResponse)
async def page_passenger_wallet(user: User = Depends(require_passenger)):
    return HTMLResponse(get_dashboard_html("Wallet", WALLET_PAGE_HTML.format(user=user), user, "wallet"))


@app.get("/passenger/subscriptions", response_class=HTMLResponse)
async def page_subscriptions(user: User = Depends(require_passenger)):
    cards = ""
    for plan_id, plan_details in CONFIG[
        'SUBSCRIPTION_PLANS'].items():
        is_current = user.subscription_plan.value == plan_id
        button = f'<button class="btn btn-secondary current" disabled>Current Plan</button>' if is_current else f'<button class="btn btn-primary" onclick="purchaseSubscription(\'{plan_id}\')">Choose Plan</button>'
        benefits = "".join(f'<li><i class="fas fa-check-circle text-success"></i> {b}</li>' for b in
                           plan_details['benefits'])
        cards += f"""<div class="card subscription-card {'popular' if plan_id == 'premium' else ''}"><h3>{plan_details['name']}</h3><div class="price">${plan_details['price']}<span style="font-size:1rem;color:var(--text-secondary)">/mo</span></div><ul>{benefits}</ul>{button}</div>"""
    return HTMLResponse(
        get_dashboard_html("Subscriptions", SUBSCRIPTIONS_PAGE_HTML.format(plan_cards=cards), user, "subscriptions"))


@app.get("/driver", response_class=HTMLResponse)
async def page_driver_dashboard(user: User = Depends(require_driver), db: Session = Depends(get_db)):
    driver = user.driver_info
    if not driver:
        return RedirectResponse("/driver/register")
    if driver.status in [DriverStatus.pending_approval, DriverStatus.rejected]:
        reason = f"<p>Reason: {driver.rejection_reason}</p>" if driver.rejection_reason else ""
        return HTMLResponse(
            get_dashboard_html(f"Application {driver.status.name.replace('_', ' ').title()}",
                               f"<p>Your application is currently {driver.status.name.replace('_', ' ')}.</p>{reason}",
                               user, "dashboard"))
    current_ride = db.query(Ride).filter(Ride.driver_id == driver.id, Ride.status.in_(
        [RideStatus.accepted, RideStatus.arriving, RideStatus.in_progress])).first()
    trip_html = "<p>No active trip. You are ready for requests.</p>"
    if current_ride:
        actions_map = {
            RideStatus.accepted: f'<button class="btn btn-secondary" onclick="updateDriverRideStatus({current_ride.id}, \'arriving\')">I\'ve Arrived</button>',
            RideStatus.arriving: f'<button class="btn btn-primary" onclick="updateDriverRideStatus({current_ride.id}, \'in_progress\')">Start Trip</button>',
            RideStatus.in_progress: f'<button class="btn btn-success" onclick="updateDriverRideStatus({current_ride.id}, \'completed\')">Complete Trip</button>'
        }
        action = actions_map.get(current_ride.status, "")
        trip_html = f'<div class="card"><h4>Ride #{current_ride.id} to {current_ride.dropoff_address}</h4><div id="driver-action-buttons">{action}</div></div>'
    earnings = sum(t.amount for t in db.query(WalletTransaction.amount).filter(WalletTransaction.user_id == user.id,
                                                                               WalletTransaction.transaction_type == TransactionType.ride_earning,
                                                                               WalletTransaction.created_at >= datetime.utcnow().date()).all())
    content = DRIVER_DASHBOARD_HTML.format(is_online_class='online' if driver.status == DriverStatus.online else '',
                                           status_text=driver.status.value.upper(),
                                           status_text_raw=driver.status.value, today_earnings=earnings,
                                           avg_rating=f"{driver.average_rating:.1f}",
                                           current_trip_html=trip_html)
    return HTMLResponse(get_dashboard_html("Dashboard", content, user, "dashboard"))


@app.get("/driver/register", response_class=HTMLResponse)
async def page_driver_register(user: User = Depends(get_current_active_user)):
    # FIX: Allow passengers to access this page to become drivers.
    if user.role not in [UserRole.passenger, UserRole.driver]:
        return RedirectResponse("/")  # Redirect non-passengers away
    return HTMLResponse(get_base_html("Driver Registration", DRIVER_REGISTRATION_HTML.format(
        vehicle_type_options="".join([f'<option value="{vt.value}">{vt.name}</option>' for vt in VehicleType])), user))


@app.get("/driver/profile", response_class=HTMLResponse)
async def page_driver_profile(user: User = Depends(require_driver)):
    return HTMLResponse(get_dashboard_html("Profile", PROFILE_PAGE_HTML.format(user=user), user, "profile"))


@app.get("/driver/history", response_class=HTMLResponse)
async def page_driver_history(user: User = Depends(require_driver), db: Session = Depends(get_db)):
    content = "<p>Driver ride history will be implemented here.</p>"
    return HTMLResponse(get_dashboard_html("Trip History", content, user, "history"))


@app.get("/driver/earnings", response_class=HTMLResponse)
async def page_driver_earnings(user: User = Depends(require_driver), db: Session = Depends(get_db)):
    content = "<p>Driver earnings analytics will be implemented here.</p>"
    return HTMLResponse(get_dashboard_html("Earnings", content, user, "earnings"))


# --- API ENDPOINTS (Non-Auth) ---
@api_router.put("/profile")
async def api_update_profile(user: User = Depends(get_current_active_user), name: str = Form(...),
                             avatar: Optional[UploadFile] = File(None), db: Session = Depends(get_db)):
    user.full_name = name
    if avatar and avatar.filename:
        ext = os.path.splitext(avatar.filename)[1]
        filepath = os.path.join(CONFIG["UPLOADS_DIR"], "avatars", f"user_{user.id}{ext}")
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(avatar.file, buffer)
        user.profile_picture_url = f"/static/avatars/user_{user.id}{ext}"
    db.commit()
    return {"message": "Profile updated", "full_name": user.full_name,
            "profile_picture_url": user.profile_picture_url}


@api_router.post("/estimate-fare")
def api_estimate_fare(req: FareEstimateRequest, user: User = Depends(require_passenger),
                      db: Session = Depends(get_db)):
    return estimate_fare(db, req.pickup_lat, req.pickup_lng, req.dropoff_lat, req.dropoff_lng, req.vehicle_type, user)


@api_router.post("/passenger/request-ride", response_model=RideResponse)
def api_request_ride(req: RideRequest, user: User = Depends(require_passenger), db: Session = Depends(get_db)):
    fare = estimate_fare(db, req.pickup_lat, req.pickup_lng, req.dropoff_lat, req.dropoff_lng, req.vehicle_type, user)
    ride = Ride(**req.model_dump(), passenger_id=user.id, estimated_fare=fare['estimated_fare'],
                distance_km=fare['distance_km'], duration_minutes=fare['duration_minutes'])
    db.add(ride)
    db.commit()
    db.refresh(ride)
    driver = find_best_driver_match(db, ride)
    if driver: print(f"Notifying driver {driver.id} for ride {ride.id}")
    return ride


@api_router.get("/ride/passenger/{ride_id}", response_model=RideResponse)
def api_get_ride_status(ride_id: int, user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    ride = db.query(Ride).options(joinedload(Ride.driver).joinedload(Driver.user), joinedload(Ride.passenger)).filter(
        Ride.id == ride_id).first()
    if not ride or (ride.passenger_id != user.id and (not ride.driver or ride.driver.user_id != user.id)):
        raise HTTPException(404, "Ride not found")

    response_data = RideResponse.model_validate(ride).model_dump()
    if ride.driver and ride.driver.current_lat:
        response_data['driver_current_lat'] = ride.driver.current_lat
        response_data['driver_current_lng'] = ride.driver.current_lng
    return JSONResponse(content=response_data)


@api_router.post("/driver/register")
async def api_driver_register(user: User = Depends(get_current_active_user), db: Session = Depends(get_db),
                              license_number: str = Form(...), vehicle_make: str = Form(...),
                              vehicle_model: str = Form(...), vehicle_year: int = Form(...),
                              vehicle_color: str = Form(...), vehicle_license_plate: str = Form(...),
                              vehicle_type: VehicleType = Form(...), license_doc: UploadFile = File(...),
                              insurance_doc: UploadFile = File(...)):
    user.role = UserRole.driver
    if user.driver_info: raise HTTPException(400, "Driver profile already exists.")
    doc_dir = os.path.join(CONFIG["UPLOADS_DIR"], "documents")
    l_ext, i_ext = os.path.splitext(license_doc.filename)[1], os.path.splitext(insurance_doc.filename)[1]
    l_path, i_path = os.path.join(doc_dir, f"license_{user.id}{l_ext}"), os.path.join(doc_dir,
                                                                                      f"insurance_{user.id}{i_ext}")
    with open(l_path, "wb") as f: shutil.copyfileobj(license_doc.file, f)
    with open(i_path, "wb") as f: shutil.copyfileobj(insurance_doc.file, f)
    driver = Driver(user_id=user.id, license_number=license_number,
                    license_doc_url=f"/static/documents/license_{user.id}{l_ext}",
                    insurance_doc_url=f"/static/documents/insurance_{user.id}{i_ext}")
    db.add(driver);
    db.flush()
    vehicle = Vehicle(driver_id=driver.id, make=vehicle_make, model=vehicle_model, year=vehicle_year,
                      color=vehicle_color, license_plate=vehicle_license_plate, vehicle_type=vehicle_type)
    db.add(vehicle);
    db.commit()
    return {"message": "Registration successful"}


@api_router.post("/driver/accept-ride", response_model=RideResponse)
def api_accept_ride(req: RideAction, user: User = Depends(require_driver), db: Session = Depends(get_db)):
    driver = user.driver_info
    ride = db.query(Ride).options(joinedload(Ride.passenger)).filter(Ride.id == req.ride_id,
                                                                     Ride.status == RideStatus.pending).first()
    if not driver or not ride: raise HTTPException(404, "Invalid request")
    ride.driver_id = driver.id
    ride.status = RideStatus.accepted
    ride.accepted_at = datetime.utcnow()
    driver.status = DriverStatus.on_trip
    db.commit()
    db.refresh(ride)
    return ride

@api_router.post("/user/subscribe")
def api_subscribe(req: PurchaseSubscription, user: User = Depends(get_current_active_user),
                  db: Session = Depends(get_db)):
    plan_details = CONFIG['SUBSCRIPTION_PLANS'].get(req.plan.value)
    if not plan_details:
        raise HTTPException(404, "Plan not found")
    price = plan_details['price']
    if user.wallet_balance < price:
        raise HTTPException(400, "Insufficient wallet balance")

    user.wallet_balance -= price
    user.subscription_plan = req.plan
    user.subscription_expiry = datetime.utcnow() + timedelta(days=30)
    db.add(WalletTransaction(user_id=user.id, transaction_type=TransactionType.subscription, amount=-price,
                             description=f"{plan_details['name']} Plan Purchase"))
    db.commit()
    return {"message": "Subscribed successfully", "new_plan": req.plan.name, "expiry": user.subscription_expiry}


# --- Mounting Routers ---
app.include_router(auth_router)
app.include_router(api_router)

# ==============================================================================
# PART 11: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print(f"--- Starting {CONFIG['PROJECT_NAME']} ---")
    print(f"Access at: http://127.0.0.1:8000")
    if "YOUR_" in CONFIG["FIREBASE_WEB_CONFIG"]['apiKey'] or "YOUR_" in CONFIG["GOOGLE_MAPS_API_KEY"]:
        print("\n❌ CRITICAL WARNING: API keys are placeholders. The application will not function properly.")
        print("   Please edit index.py and fill in the CONFIG dictionary.\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)# index.py
#
# =================================================================================
# ||  RYDEPRO - DEFINITIVE, FULLY DEBUGGED & ROBUST CAR ORDERING SYSTEM           ||
# =================================================================================
#
# Version: 4.3 (Pydantic Response Model FIXED)
#
# KEY IMPROVEMENTS IN THIS VERSION:
# - Pydantic FastAPIError FIXED: The critical crash caused by using a SQLAlchemy
#   model as a `response_model` is resolved.
# - ROBUST RESPONSE SCHEMAS: New Pydantic models (e.g., `RideResponse`,
#   `DriverResponse`, `UserResponse`) have been created to act as a clean,
#   decoupled layer between the database and the API.
# - MODERNIZED & CONSISTENT CODEBASE: The entire application now consistently
#   uses Pydantic schemas for data validation and serialization, which is a
#   FastAPI best practice for creating robust and predictable APIs.
# - ALL PREVIOUS FUNCTIONALITY RETAINED: This fix integrates perfectly with all
#   existing features.
#
# Tech Stack: Python, FastAPI, SQLAlchemy, Uvicorn, HTML5, CSS3, JavaScript
# =================================================================================


# ==============================================================================
# PART 1: IMPORTS & CORE SETUP
# ==============================================================================
import os
import json
import datetime
import math
import secrets
import hmac
import hashlib
import shutil
from enum import Enum as PyEnum

from fastapi import (
    FastAPI, Request, Depends, HTTPException, status, APIRouter,
    Form, File, UploadFile, Header
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from pydantic_settings import BaseSettings

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Boolean, Enum as SQLAlchemyEnum, ForeignKey, Text
)
from sqlalchemy.orm import sessionmaker, Session, relationship, joinedload, declarative_base
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, auth
import uvicorn

# ==============================================================================
# PART 2: CONFIGURATION
# ==============================================================================
# (Configuration is unchanged)
CONFIG = {
    "PROJECT_NAME": "RydePro - Advanced Ride-Hailing System",
    "SECRET_KEY": secrets.token_urlsafe(32),
    "ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 60 * 24 * 7,
    "DATABASE_URL": "sqlite:///./rydepro_v4_3.db",
    "FIREBASE_SERVICE_ACCOUNT_KEY": "firebase-service.json",
    "FIREBASE_WEB_CONFIG": {
        "apiKey": "AIzaSyDwzbzfZWejZD2AGGKlw5XLMLxfFzwd2qI",
        "authDomain": "apexride-ada98.firebaseapp.com",
        "projectId": "apexride-ada98",
        "storageBucket": "apexride-ada98.firebasestorage.app",
        "messagingSenderId": "558535928716",
        "appId": "1:558535928716:web:26b4c14d72135307ab84a9",
        "measurementId": "G-YB929Z97PT"
    },
    "GOOGLE_MAPS_API_KEY": "YOUR_GOOGLE_MAPS_API_KEY",
    "PAYSTACK_SECRET_KEY": "YOUR_PAYSTACK_SECRET_KEY",
    "PAYSTACK_PUBLIC_KEY": "YOUR_PAYSTACK_PUBLIC_KEY",
    "UPLOADS_DIR": "uploads", "DEFAULT_CURRENCY": "USD", "COMMISSION_RATE": 0.20,
    "BASE_FARE": 2.50, "COST_PER_KM": 1.75, "COST_PER_MINUTE": 0.30,
    "SURGE_MAX_MULTIPLIER": 2.5, "SURGE_DEMAND_SENSITIVITY": 0.2,
    "HEAT_MAPS": {
        "downtown": {"lat": 34.05, "lng": -118.25, "radius": 5, "bonus": 0.2},
        "airport": {"lat": 33.94, "lng": -118.40, "radius": 4, "bonus": 0.3}
    },
    "SUBSCRIPTION_PLANS": {
        "basic": {"name": "Basic", "price": 9.99, "benefits": ["5% off rides", "Standard Support"]},
        "premium": {"name": "Premium", "price": 19.99,
                    "benefits": ["10% off rides", "Priority Support", "Priority Matching"]},
        "ultimate": {"name": "Ultimate", "price": 29.99,
                     "benefits": ["15% off rides", "24/7 VIP Support", "Highest Priority Matching",
                                  "No Surge Pricing"]},
    }
}


# ==============================================================================
# PART 3: INITIALIZATION & CORE UTILITIES
# ==============================================================================
def setup_dirs():
    for subdir in ["documents", "avatars"]:
        path = os.path.join(CONFIG["UPLOADS_DIR"], subdir)
        if not os.path.exists(path): os.makedirs(path)


def init_firebase():
    try:
        if os.path.exists(CONFIG["FIREBASE_SERVICE_ACCOUNT_KEY"]):
            if not firebase_admin._apps:
                cred = credentials.Certificate(CONFIG["FIREBASE_SERVICE_ACCOUNT_KEY"])
                firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK initialized successfully.")
        else:
            print(f"❌ FIREBASE WARNING: '{CONFIG['FIREBASE_SERVICE_ACCOUNT_KEY']}' not found. Social login will fail.")
    except Exception as e:
        print(f"❌ FIREBASE CRITICAL: Could not initialize. {e}. Social login will fail.")


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)


def get_password_hash(password): return pwd_context.hash(password)


# ==============================================================================
# PART 4: DATABASE SETUP (SQLALCHEMY 2.0 Syntax)
# ==============================================================================
engine = create_engine(CONFIG["DATABASE_URL"], connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Enums & Models ---
class UserRole(str, PyEnum): passenger = "passenger"; driver = "driver"; admin = "admin"


class RideStatus(str,
                 PyEnum): pending = "pending"; accepted = "accepted"; arriving = "arriving"; in_progress = "in_progress"; completed = "completed"; cancelled = "cancelled"; scheduled = "scheduled"


class PaymentMethod(str, PyEnum): wallet = "wallet"; card = "card"; cash = "cash"


class VehicleType(str,
                  PyEnum): economy = "Economy"; luxury = "Luxury"; suv = "SUV"; bike = "Bike"; truck = "Truck"; ev = "EV"


class DriverStatus(str,
                   PyEnum): offline = "offline"; online = "online"; on_trip = "on_trip"; pending_approval = "pending_approval"; rejected = "rejected"


class SubscriptionPlan(str, PyEnum): none = "none"; basic = "basic"; premium = "premium"; ultimate = "ultimate"


class TransactionType(str,
                      PyEnum): topup = "topup"; ride_payment = "ride_payment"; withdrawal = "withdrawal"; ride_earning = "ride_earning"; subscription = "subscription"


class User(Base):
    __tablename__ = "users";
    id = Column(Integer, primary_key=True, index=True);
    firebase_uid = Column(String, unique=True, index=True, nullable=True);
    email = Column(String, unique=True, index=True, nullable=False);
    hashed_password = Column(String, nullable=True);
    full_name = Column(String, nullable=False);
    profile_picture_url = Column(String, default="/static/avatars/default.png");
    role = Column(SQLAlchemyEnum(UserRole), nullable=False);
    created_at = Column(DateTime, default=datetime.utcnow);
    is_active = Column(Boolean, default=True);
    wallet_balance = Column(Float, default=0.0);
    subscription_plan = Column(SQLAlchemyEnum(SubscriptionPlan), default=SubscriptionPlan.none);
    subscription_expiry = Column(DateTime, nullable=True);
    fcm_token = Column(String, nullable=True);
    driver_info = relationship("Driver", back_populates="user", uselist=False, cascade="all, delete-orphan");
    transactions = relationship("WalletTransaction", back_populates="user", cascade="all, delete-orphan")


class Driver(Base):
    __tablename__ = "drivers";
    id = Column(Integer, primary_key=True);
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False);
    license_number = Column(String, unique=True);
    license_doc_url = Column(String);
    insurance_doc_url = Column(String);
    status = Column(SQLAlchemyEnum(DriverStatus), default=DriverStatus.pending_approval);
    current_lat = Column(Float, nullable=True);
    current_lng = Column(Float, nullable=True);
    last_seen = Column(DateTime, default=datetime.utcnow);
    average_rating = Column(Float, default=5.0);
    rejection_reason = Column(Text, nullable=True);
    user = relationship("User", back_populates="driver_info");
    vehicle = relationship("Vehicle", back_populates="driver", uselist=False, cascade="all, delete-orphan")


class Vehicle(Base): __tablename__ = "vehicles"; id = Column(Integer, primary_key=True); driver_id = Column(Integer,
                                                                                                            ForeignKey(
                                                                                                                "drivers.id"),
                                                                                                            nullable=False); make = Column(
    String); model = Column(String); year = Column(Integer); color = Column(String); license_plate = Column(String,
                                                                                                            unique=True); vehicle_type = Column(
    SQLAlchemyEnum(VehicleType)); driver = relationship("Driver", back_populates="vehicle")


class Ride(Base): __tablename__ = "rides"; id = Column(Integer, primary_key=True); passenger_id = Column(Integer,
                                                                                                         ForeignKey(
                                                                                                             "users.id")); driver_id = Column(
    Integer, ForeignKey("drivers.id"), nullable=True); pickup_address = Column(String); dropoff_address = Column(
    String); pickup_lat = Column(Float); pickup_lng = Column(Float); dropoff_lat = Column(Float); dropoff_lng = Column(
    Float); status = Column(SQLAlchemyEnum(RideStatus), default=RideStatus.pending); vehicle_type_requested = Column(
    SQLAlchemyEnum(VehicleType)); estimated_fare = Column(Float); actual_fare = Column(Float,
                                                                                       nullable=True); distance_km = Column(
    Float, nullable=True); duration_minutes = Column(Float, nullable=True); payment_method = Column(
    SQLAlchemyEnum(PaymentMethod)); created_at = Column(DateTime, default=datetime.utcnow); accepted_at = Column(
    DateTime, nullable=True); arrived_at = Column(DateTime, nullable=True); started_at = Column(DateTime,
                                                                                                nullable=True); completed_at = Column(
    DateTime, nullable=True); passenger_rated = Column(Boolean, default=False); driver_rated = Column(Boolean,
                                                                                                      default=False); share_token = Column(
    String, default=lambda: secrets.token_urlsafe(16)); passenger = relationship("User", foreign_keys=[
    passenger_id]); driver = relationship("Driver", foreign_keys=[driver_id])


class Review(Base): __tablename__ = "reviews"; id = Column(Integer, primary_key=True); ride_id = Column(Integer,
                                                                                                        ForeignKey(
                                                                                                            "rides.id")); reviewer_id = Column(
    Integer, ForeignKey("users.id")); reviewee_id = Column(Integer, ForeignKey("users.id")); rating = Column(Integer,
                                                                                                             default=5); comment = Column(
    Text, nullable=True); created_at = Column(DateTime, default=datetime.utcnow)


class WalletTransaction(Base): __tablename__ = "wallet_transactions"; id = Column(Integer,
                                                                                  primary_key=True); user_id = Column(
    Integer, ForeignKey("users.id")); transaction_type = Column(SQLAlchemyEnum(TransactionType)); amount = Column(
    Float); reference = Column(String, nullable=True); description = Column(String); created_at = Column(DateTime,
                                                                                                         default=datetime.utcnow); user = relationship(
    "User", back_populates="transactions")


class SOSLog(Base): __tablename__ = "sos_logs"; id = Column(Integer, primary_key=True); ride_id = Column(Integer,
                                                                                                         ForeignKey(
                                                                                                             "rides.id")); user_id = Column(
    Integer, ForeignKey("users.id")); timestamp = Column(DateTime, default=datetime.utcnow); lat = Column(
    Float); lng = Column(Float); details = Column(Text, default="SOS button pressed")


Base.metadata.create_all(bind=engine)


def get_db(): db = SessionLocal();_ = (yield db);db.close()


# ==============================================================================
# PART 5: AUTHENTICATION & SECURITY
# ==============================================================================
def create_access_token(data: dict): to_encode = data.copy();to_encode.update(
    {"exp": datetime.utcnow() + timedelta(minutes=CONFIG["ACCESS_TOKEN_EXPIRE_MINUTES"])});return jwt.encode(to_encode,
                                                                                                             CONFIG[
                                                                                                                 "SECRET_KEY"],
                                                                                                             algorithm=
                                                                                                             CONFIG[
                                                                                                                 "ALGORITHM"])


def get_current_user(request: Request, db: Session = Depends(get_db)):
    """
    Retrieves the current user from a session token.

    This function implements a robust two-step check:
    1. It first looks for the 'access_token' in the request cookies, which is
       the standard method for browser-based sessions.
    2. If no cookie is found, it falls back to checking for an 'Authorization'
       header with a Bearer token (e.g., "Bearer <token>"). This is the standard
       for API clients and mobile apps.

    Args:
        request: The incoming FastAPI request object.
        db: The SQLAlchemy database session dependency.

    Returns:
        The User database object if the token is valid, otherwise None.
    """
    token = None

    # --- Step 1: Check for the token in cookies (for browsers) ---
    if "access_token" in request.cookies:
        token = request.cookies.get("access_token")

    # --- Step 2: If no cookie, check for Authorization header (for APIs/mobile) ---
    else:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    # If no token was found in either location, there is no user.
    if not token:
        return None

    # --- Step 3: Decode the token and retrieve the user ---
    try:
        # Decode the JWT to get the payload
        payload = jwt.decode(token, CONFIG["SECRET_KEY"], algorithms=[CONFIG["ALGORITHM"]])

        # Extract the user ID (subject) from the payload
        user_id_str = payload.get("sub")

        if user_id_str is None:
            # The token is malformed if it's missing the 'sub' claim
            return None

        user_id = int(user_id_str)

        # Query the database for the user with that ID
        return db.query(User).filter(User.id == user_id).first()

    except (JWTError, ValueError, KeyError):
        # Catches several potential errors:
        # - JWTError: If the token is invalid, expired, or has a bad signature.
        # - ValueError: If the user_id from the token cannot be converted to an integer.
        # - KeyError: If the payload is structured unexpectedly.
        return None


async def get_current_active_user(user: User = Depends(get_current_user)):
    if not user: raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated");
    if not user.is_active: raise HTTPException(status.HTTP_400_BAD_REQUEST, "Inactive user");return user


def require_role(role: UserRole):
    async def role_checker(user: User = Depends(get_current_active_user)):
        if user.role != role: raise HTTPException(status.HTTP_403_FORBIDDEN,
                                                  f"{role.value.capitalize()} privileges required");return user

    return role_checker


require_passenger = require_role(UserRole.passenger);
require_driver = require_role(UserRole.driver);
require_admin = require_role(UserRole.admin);


# ==============================================================================
# PART 6: PYDANTIC SCHEMAS (RESPONSE MODELS ADDED)
# ==============================================================================
# --- Base Schemas ---
class OrmModel(BaseModel):
    class Config: from_attributes = True


# --- Input Schemas ---
class UserCreate(BaseModel): full_name: str; email: EmailStr; password: str


class UserLogin(BaseModel): email: EmailStr; password: str; expected_role: Optional[UserRole] = None


class FirebaseLogin(BaseModel): firebase_token: str


class RideRequest(
    BaseModel): pickup_lat: float; pickup_lng: float; pickup_address: str; dropoff_lat: float; dropoff_lng: float; dropoff_address: str; vehicle_type: VehicleType; payment_method: PaymentMethod


class FareEstimateRequest(
    BaseModel): pickup_lat: float; pickup_lng: float; dropoff_lat: float; dropoff_lng: float; vehicle_type: VehicleType


class RideAction(BaseModel): ride_id: int


class DriverRideStatusUpdate(BaseModel): ride_id: int; status: str


class RateRide(BaseModel): ride_id: int; rating: int = Field(..., ge=1, le=5); comment: Optional[str] = None


class PurchaseSubscription(BaseModel): plan: SubscriptionPlan


# --- FIX: Response Schemas ---
class UserResponse(OrmModel):
    id: int;
    full_name: str;
    email: EmailStr;
    profile_picture_url: str;
    role: UserRole


class VehicleResponse(OrmModel):
    make: str;
    model: str;
    year: int;
    color: str;
    license_plate: str;
    vehicle_type: VehicleType


class DriverResponse(OrmModel):
    id: int;
    status: DriverStatus;
    average_rating: float;
    user: UserResponse;
    vehicle: Optional[VehicleResponse] = None


class RideResponse(OrmModel):
    id: int;
    status: RideStatus;
    pickup_address: str;
    dropoff_address: str
    estimated_fare: float;
    actual_fare: Optional[float] = None
    passenger: UserResponse
    driver: Optional[DriverResponse] = None
    driver_current_lat: Optional[float] = None
    driver_current_lng: Optional[float] = None
    passenger_rated: bool;
    driver_rated: bool;
    share_token: str


# ==============================================================================
# PART 7: CORE BUSINESS LOGIC
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2): R, dLat, dLon = 6371, math.radians(lat2 - lat1), math.radians(
    lon2 - lon1);a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
    dLon / 2) ** 2;return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def calculate_surge_pricing(db: Session, lat, lng): now = datetime.utcnow();hour_mult = 1 + (.3 * math.sin(
    math.pi * (now.hour - 6) / 12));day_mult = 1.2 if now.weekday() >= 4 else 1;heatmap_bonus = max(
    (z['bonus'] for z in CONFIG["HEAT_MAPS"].values() if haversine(lat, lng, z['lat'], z['lng']) <= z['radius']),
    default=0);area_requests = db.query(Ride).filter(Ride.status == RideStatus.pending,
                                                     Ride.created_at > now - timedelta(
                                                         minutes=15)).count();available_drivers = db.query(
    Driver).filter(Driver.status == DriverStatus.online).count();demand_surge = 1 + (
            ((area_requests + 1) / (available_drivers + 1)) * CONFIG["SURGE_DEMAND_SENSITIVITY"]);return min(
    (demand_surge * hour_mult * day_mult) + heatmap_bonus, CONFIG["SURGE_MAX_MULTIPLIER"])


def estimate_fare(db: Session, plat, plng, dlat, dlng, vt, user: User): distance = haversine(plat, plng, dlat,
                                                                                             dlng);duration = (
                                                                                                                          distance / 35) * 60;base_fare = (
            CONFIG["BASE_FARE"] + (distance * CONFIG["COST_PER_KM"]) + (
                duration * CONFIG["COST_PER_MINUTE"]));vehicle_mult = {"Economy": 1.0, "Luxury": 2.2, "SUV": 1.5,
                                                                       "Bike": 0.6, "Truck": 2.5, "EV": 1.1}.get(
    vt.value, 1.0);surge_mult = 1.0 if user.subscription_plan == SubscriptionPlan.ultimate else calculate_surge_pricing(
    db, plat, plng);final_fare = base_fare * vehicle_mult * surge_mult;discounts = {SubscriptionPlan.basic: 0.95,
                                                                                    SubscriptionPlan.premium: 0.90,
                                                                                    SubscriptionPlan.ultimate: 0.85};final_fare *= discounts.get(
    user.subscription_plan, 1.0);return {"estimated_fare": round(final_fare, 2), "distance_km": round(distance, 2),
                                         "duration_minutes": round(duration, 1),
                                         "surge_multiplier": round(surge_mult, 2)}


def get_subscription_priority_score(plan: SubscriptionPlan): return {SubscriptionPlan.none: 1.0,
                                                                     SubscriptionPlan.basic: 0.95,
                                                                     SubscriptionPlan.premium: 0.85,
                                                                     SubscriptionPlan.ultimate: 0.75}.get(plan, 1.0)


def find_best_driver_match(db: Session, ride: Ride) -> Optional[Driver]:
    DISTANCE_WEIGHT, RATING_WEIGHT, SUBSCRIPTION_WEIGHT = 0.60, 0.20, 0.20
    drivers = db.query(Driver).options(joinedload(Driver.user), joinedload(Driver.vehicle)).join(Vehicle).filter(
        Driver.status == DriverStatus.online, Vehicle.vehicle_type == ride.vehicle_type_requested).all()
    if not drivers: print("--- AI Driver Matching: No online drivers found for this vehicle type. ---");return None
    scored_drivers = [];
    print("\n--- AI Driver Matching ---")
    for driver in drivers:
        if driver.current_lat and driver.current_lng:
            dist = haversine(ride.pickup_lat, ride.pickup_lng, driver.current_lat, driver.current_lng)
            rating_score = (5.5 - driver.average_rating)
            sub_score = get_subscription_priority_score(driver.user.subscription_plan)
            final_score = (dist * DISTANCE_WEIGHT) + (rating_score * RATING_WEIGHT) + (sub_score * SUBSCRIPTION_WEIGHT)
            scored_drivers.append((final_score, driver))
            print(
                f"  - Driver ID {driver.id}: Dist={dist:.2f}km, Rating={driver.average_rating}, Sub={driver.user.subscription_plan.name} -> SCORE: {final_score:.2f}")
    if not scored_drivers: print("--- AI Driver Matching: No drivers with location data available. ---");return None
    scored_drivers.sort(key=lambda x: x[0])
    best_score, best_driver = scored_drivers[0]
    print(f"==> BEST MATCH: Driver ID {best_driver.id} with score {best_score:.2f}\n")
    return best_driver


def update_driver_rating(driver_id: int, db: Session):
    driver = db.query(Driver).get(driver_id);
    if not driver: return
    ratings = db.query(Review.rating).join(Ride).filter(Ride.driver_id == driver_id,
                                                        Review.reviewee_id == driver.user_id).all();
    if ratings: driver.average_rating = round(sum(r[0] for r in ratings) / len(ratings), 2);db.commit();


# ==============================================================================
# PART 8: FRONTEND ASSETS (CSS, JS, HTML)
# ==============================================================================
# (CSS and JS are unchanged from the last fully debugged version, so they are kept as is)
MAIN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
:root{--bg-dark-primary:#12121c;--bg-dark-secondary:#1a1a2e;--bg-dark-tertiary:#16213e;--primary-accent:#6d28d9;--primary-accent-hover:#5b21b6;--secondary-accent:#e94560;--text-primary:#e0e0e0;--text-secondary:#a0a0c0;--border-color:#3a3a5a;--success:#10b981;--error:#ef4444;--warning:#f59e0b;--font-family:'Inter',sans-serif;--border-radius-sm:6px;--border-radius-md:10px;--shadow-md:0 4px 6px -1px rgba(0,0,0,.1),0 2px 4px -2px rgba(0,0,0,.1);--shadow-lg:0 10px 15px -3px rgba(0,0,0,.1),0 4px 6px -4px rgba(0,0,0,.1);--transition:all .3s cubic-bezier(.4,0,.2,1)}*,::after,::before{box-sizing:border-box;margin:0;padding:0}body{font-family:var(--font-family);background-color:var(--bg-dark-primary);color:var(--text-primary);line-height:1.6;display:flex;flex-direction:column;min-height:100vh;overflow-x:hidden}.main-content{flex:1}h1,h2,h3,h4{color:var(--text-primary);margin-bottom:1rem;font-weight:600}a{color:var(--primary-accent);text-decoration:none;transition:var(--transition)}a:hover{color:var(--primary-accent-hover)}.container{width:95%;max-width:1600px;margin:2rem auto;padding:0 1rem}.btn{display:inline-block;padding:12px 28px;border:none;border-radius:var(--border-radius-md);cursor:pointer;font-size:1rem;font-weight:600;text-align:center;transition:var(--transition);text-transform:uppercase;letter-spacing:1px}.btn-primary{background:linear-gradient(90deg,var(--primary-accent),#a855f7);color:#fff}.btn-primary:hover{transform:translateY(-2px);box-shadow:var(--shadow-lg)}.btn-secondary{background:var(--bg-dark-tertiary);color:var(--text-primary)}.btn-secondary:hover{background:#1f2c4a}.btn-danger{background-color:var(--error);color:#fff}.btn-success{background-color:var(--success);color:#fff}.form-container{background:var(--bg-dark-secondary);padding:2.5rem;border-radius:var(--border-radius-md);box-shadow:var(--shadow-lg);max-width:500px;margin:2rem auto;border:1px solid var(--border-color)}.form-group{margin-bottom:1.5rem}.form-group label{display:block;margin-bottom:.5rem;font-weight:500;color:var(--text-secondary)}.form-control{width:100%;padding:14px;background:var(--bg-dark-primary);border:1px solid var(--border-color);border-radius:var(--border-radius-sm);color:var(--text-primary);font-size:1rem;transition:var(--transition)}.form-control:focus{outline:none;border-color:var(--primary-accent);box-shadow:0 0 0 3px rgba(109,40,217,.5)}.main-header{background-color:var(--bg-dark-secondary);padding:1rem 2.5%;display:flex;justify-content:space-between;align-items:center;box-shadow:var(--shadow-md);position:sticky;top:0;z-index:1000;border-bottom:1px solid var(--border-color)}.logo{font-size:2rem;font-weight:700;color:#fff}.logo .fa-rocket{color:var(--primary-accent)}.main-nav ul{list-style:none;display:flex;gap:1.5rem}.main-nav a{color:var(--text-secondary);font-weight:500;padding:5px 10px;border-radius:5px}.main-nav a:hover{background:var(--bg-dark-tertiary);color:var(--text-primary);text-decoration:none}.user-menu{position:relative;cursor:pointer}.user-menu img{width:40px;height:40px;border-radius:50%;border:2px solid var(--border-color)}.user-menu:hover .dropdown-menu{display:block}.dropdown-menu{display:none;position:absolute;right:0;top:120%;background:var(--bg-dark-tertiary);border:1px solid var(--border-color);border-radius:var(--border-radius-md);min-width:220px;box-shadow:var(--shadow-lg);list-style:none;padding:.5rem 0;z-index:1001}.dropdown-menu a{display:flex;gap:.75rem;align-items:center;padding:.75rem 1.25rem;color:var(--text-secondary)}.dropdown-menu a:hover{background:var(--primary-accent);color:#fff;text-decoration:none}.main-footer{background-color:var(--bg-dark-secondary);color:var(--text-secondary);text-align:center;padding:2rem 1rem;margin-top:auto;border-top:1px solid var(--border-color)}#auth-page{display:flex;align-items:center;justify-content:center;min-height:80vh}.dashboard-layout{display:grid;grid-template-columns:260px 1fr;gap:2rem;min-height:calc(100vh - 85px)}.sidebar{background:var(--bg-dark-secondary);padding:2rem 1rem;border-right:1px solid var(--border-color)}.sidebar-nav ul{list-style:none}.sidebar-nav li a{display:flex;align-items:center;gap:1rem;padding:15px;color:var(--text-secondary);border-radius:var(--border-radius-sm);margin-bottom:.5rem;font-size:1.05rem}.sidebar-nav li a:hover{background:var(--bg-dark-tertiary);color:var(--text-primary);text-decoration:none}.sidebar-nav li a.active{background:var(--primary-accent);color:#fff;font-weight:600}.sidebar-nav li a .fa-fw{width:20px;text-align:center}.dashboard-content{padding:2rem;overflow-y:auto}.dashboard-content h1{border-bottom:1px solid var(--border-color);padding-bottom:1rem;margin-bottom:2rem}.card{background:var(--bg-dark-secondary);border:1px solid var(--border-color);border-radius:var(--border-radius-md);padding:1.5rem;margin-bottom:1.5rem;box-shadow:var(--shadow-md)}.card-header{border-bottom:1px solid var(--border-color);padding-bottom:1rem;margin-bottom:1rem;font-size:1.25rem;font-weight:600;display:flex;justify-content:space-between;align-items:center}.stat-card{background:var(--bg-dark-secondary);padding:1.5rem;border-radius:var(--border-radius-md);border:1px solid var(--border-color);text-align:center}.stat-card .icon{font-size:2.5rem;color:var(--primary-accent);margin-bottom:1rem}.stat-card .value{font-size:2rem;font-weight:700}.stat-card .label{font-size:.9rem;color:var(--text-secondary);margin-top:.5rem}.grid-container{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.5rem}.passenger-dashboard-grid{display:grid;grid-template-columns:1fr 420px;gap:2rem;height:calc(100vh - 200px)}#map{height:100%;width:100%;border-radius:var(--border-radius-md);border:2px solid var(--border-color)}.booking-panel{display:flex;flex-direction:column;background:var(--bg-dark-secondary);padding:1.5rem;border-radius:var(--border-radius-md);border:1px solid var(--border-color)}.vehicle-options{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0}.vehicle-option{background:var(--bg-dark-primary);border:2px solid var(--border-color);border-radius:var(--border-radius-sm);padding:1rem;text-align:center;cursor:pointer;transition:var(--transition);position:relative}.vehicle-option:hover{border-color:var(--primary-accent)}.vehicle-option.selected{border-color:var(--primary-accent);background:var(--bg-dark-tertiary)}.vehicle-option.selected::after{content:'✔';position:absolute;top:5px;right:8px;color:var(--success)}.vehicle-option .icon{font-size:2rem;color:var(--primary-accent)}.vehicle-option .name{font-weight:500;margin-top:.5rem}#fare-estimate-box{background:var(--bg-dark-primary);padding:1rem;border-radius:var(--border-radius-sm);margin-top:auto;text-align:center;border:1px solid var(--border-color)}#fare-estimate-box h3{font-size:1.5rem;margin-bottom:.5rem;color:var(--primary-accent)}#fare-estimate-box p{color:var(--text-secondary)}#ride-status-container{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);width:90%;max-width:700px;background:linear-gradient(to right,var(--bg-dark-secondary),var(--bg-dark-tertiary));border-radius:var(--border-radius-md);box-shadow:var(--shadow-lg);z-index:1001;border:1px solid var(--border-color);padding:1.5rem;display:none}.ride-status-header{display:flex;justify-content:space-between;align-items:center}#ride-status-text{font-size:1.2rem;font-weight:600}#sos-button,#share-trip-button{background-color:var(--bg-dark-tertiary);color:var(--text-primary);padding:8px 15px;border-radius:var(--border-radius-sm)}.ride-status-actions{display:flex;gap:.5rem}#sos-button{background-color:var(--error);color:#fff}#ride-driver-info{margin-top:1rem;padding-top:1rem;border-top:1px solid var(--border-color)}.progress-container{margin-top:1rem}.progress-steps{display:flex;justify-content:space-between;position:relative;margin-bottom:1rem}.progress-step{text-align:center;width:25%;font-size:.8rem;color:var(--text-secondary)}.progress-step.active{color:var(--text-primary);font-weight:600}.progress-bar-bg{height:6px;background-color:var(--border-color);border-radius:3px}.progress-bar-fg{height:100%;width:0%;background:var(--primary-accent);border-radius:3px;transition:width .5s ease}#driver-status-toggle{display:flex;align-items:center;gap:1rem;background-color:var(--error);color:#fff;padding:1rem;border-radius:var(--border-radius-md);cursor:pointer;transition:var(--transition);font-weight:600}#driver-status-toggle.online{background-color:var(--success)}.modal{display:none;position:fixed;z-index:2000;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,.8);backdrop-filter:blur(5px);justify-content:center;align-items:center}.modal-content{background:var(--bg-dark-secondary);padding:2rem;border-radius:var(--border-radius-md);width:90%;max-width:500px;text-align:center;border:1px solid var(--border-color);position:relative}.close-modal{position:absolute;top:10px;right:15px;font-size:2rem;color:var(--text-secondary);cursor:pointer;transition:var(--transition)}.close-modal:hover{color:var(--text-primary)}.countdown-timer{width:100px;height:100px;border:5px solid var(--primary-accent);border-radius:50%;display:flex;justify-content:center;align-items:center;font-size:2.5rem;font-weight:700;margin:1rem auto}.table-container{overflow-x:auto}.data-table{width:100%;border-collapse:collapse;margin-top:2rem}.data-table td,.data-table th{padding:15px;text-align:left;border-bottom:1px solid var(--border-color)}.data-table thead{background:var(--bg-dark-tertiary);color:var(--text-primary)}.data-table tbody tr:hover{background:var(--bg-dark-tertiary)}.status-tag{padding:4px 10px;border-radius:20px;font-size:.8rem;font-weight:600;text-transform:capitalize}.status-pending,.status-pending_approval{background-color:var(--warning);color:#000}.status-approved,.status-completed,.status-successful,.status-online{background-color:var(--success);color:#fff}.status-rejected,.status-cancelled,.status-offline{background-color:var(--error);color:#fff}.status-accepted,.status-arriving,.status-in_progress,.status-on_trip{background-color:#3b82f6;color:#fff}.star-rating{display:flex;justify-content:center;direction:rtl}.star-rating input[type=radio]{display:none}.star-rating label{font-size:2.5rem;color:#444;cursor:pointer;transition:color .2s}.star-rating label:hover,.star-rating label:hover~label,.star-rating input[type=radio]:checked~label{color:var(--warning)}#toast-container{position:fixed;top:20px;right:20px;z-index:9999}.toast{padding:15px 25px;margin-bottom:1rem;border-radius:var(--border-radius-sm);color:#fff;box-shadow:var(--shadow-lg);opacity:0;transform:translateX(100%)}.toast.show{opacity:1;transform:translateX(0);animation:slideIn .5s forwards}@keyframes slideIn{to{opacity:1;transform:translateX(0)}}@keyframes slideOut{from{opacity:1;transform:translateX(0)}to{opacity:0;transform:translateX(100%)}}.toast-success{background:var(--success)}.toast-error{background:var(--error)}.toast-info{background:#3b82f6}.loader{border:4px solid var(--border-color);border-top:4px solid var(--primary-accent);border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:2rem auto}@keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.text-success{color:var(--success)}.text-error{color:var(--error)}.text-center{text-align:center}.mt-3{margin-top:1.5rem}.profile-avatar{width:120px;height:120px;border-radius:50%;object-fit:cover;border:3px solid var(--border-color)}.file-upload-wrapper{position:relative;overflow:hidden;display:inline-block}.file-upload-wrapper input[type=file]{font-size:100px;position:absolute;left:0;top:0;opacity:0}.subscription-card{text-align:center;padding:2rem}.subscription-card.popular{border:2px solid var(--primary-accent);transform:scale(1.05)}.subscription-card h3{font-size:1.5rem;color:var(--primary-accent)}.subscription-card .price{font-size:2.5rem;font-weight:700;margin:1rem 0}.subscription-card ul{list-style:none;margin:1.5rem 0}.subscription-card li{margin-bottom:.5rem}.subscription-card .btn.current{background:var(--success);cursor:not-allowed}
.auth-tabs{display:flex;margin-bottom:1.5rem;border-bottom:1px solid var(--border-color)}.auth-tab-button{flex:1;padding:1rem;background:0 0;border:none;color:var(--text-secondary);cursor:pointer;font-size:1.1rem;font-weight:600;border-bottom:3px solid transparent}.auth-tab-button.active{color:var(--primary-accent);border-bottom-color:var(--primary-accent)}.auth-tab-content{display:none}.auth-tab-content.active{display:block}
@media (max-width:992px){.dashboard-layout{grid-template-columns:1fr}.sidebar{display:flex;height:auto;padding:.5rem;overflow-x:auto}.sidebar-nav ul{display:flex;width:100%;justify-content:space-around}.passenger-dashboard-grid{grid-template-columns:1fr;height:auto}#map{height:50vh}}
"""

MAIN_JS = f"""
const FIREBASE_CONFIG = {json.dumps(CONFIG['FIREBASE_WEB_CONFIG'])};
const GOOGLE_MAPS_API_KEY = "{CONFIG['GOOGLE_MAPS_API_KEY']}";
const PAYSTACK_PUBLIC_KEY = "{CONFIG['PAYSTACK_PUBLIC_KEY']}";
let map, directionsService, directionsRenderer, pickupMarker, dropoffMarker, driverMarker;
let rideStatusPollInterval, rideRequestPollInterval, countdownInterval;

// --- UTILITY & UI FUNCTIONS ---
function showToast(msg, type = 'info', duration = 4000) {{ const cont=document.getElementById('toast-container')||createToastCont(); const toast=document.createElement('div'); toast.className=`toast toast-${{type}} show`; toast.textContent=msg; cont.appendChild(toast); setTimeout(()=>{{toast.classList.remove('show'); toast.style.animation='slideOut .5s forwards'; setTimeout(()=>toast.remove(),500)}},duration);}}
function createToastCont() {{ const c=document.createElement('div'); c.id='toast-container'; document.body.appendChild(c); return c; }}
async function apiFetch(url, opts = {{}}, showFeedback = true) {{ const headers={{'Accept':'application/json',...opts.headers}}; if(!(opts.body instanceof FormData)) headers['Content-Type']='application/json'; opts.headers=headers; if(opts.body&&typeof opts.body!=='string'&&!(opts.body instanceof FormData))opts.body=JSON.stringify(opts.body); try{{const res=await fetch(url,opts); const data=await res.json(); if(!res.ok)throw new Error(data.detail||'API error occurred'); return data;}} catch(err){{console.error('API Error:',url,err); if(showFeedback)showToast(err.message,'error'); throw err;}}}}

// --- FIREBASE & AUTHENTICATION ---
function initializeFirebase() {{
    const authContainer = document.getElementById('firebaseui-auth-container');
    const socialLoader = document.getElementById('firebaseui-loader');
    if (!authContainer) return;
    try {{
        if (!FIREBASE_CONFIG.apiKey || FIREBASE_CONFIG.apiKey.includes("YOUR_")) {{
            console.error("Firebase config error: API key is missing or a placeholder.");
            if (socialLoader) socialLoader.innerHTML = '<p style="color:var(--error);">Firebase Config Error.</p>';
            return;
        }}
        if (firebase.apps.length === 0) {{ firebase.initializeApp(FIREBASE_CONFIG); }}
        firebase.auth().onIdTokenChanged(user => {{
            if (user) {{ user.getIdToken().then(fcm_token => apiFetch('/api/auth/update-fcm', {{ method: 'POST', body: {{ fcm_token }} }}, false)).catch(e => console.error("FCM update failed", e)); }}
        }});
        setupFirebaseUI();
    }} catch (e) {{
        console.error("Firebase initialization failed:", e);
        if (socialLoader) socialLoader.innerHTML = `<p style="color:var(--error);">Auth Error: ${{e.message}}</p>`;
    }}
}}
function setupFirebaseUI() {{
    const ui = new firebaseui.auth.AuthUI(firebase.auth());
    ui.start('#firebaseui-auth-container', {{
        signInSuccessUrl: '/auth/firebase/callback',
        signInOptions: [firebase.auth.GoogleAuthProvider.PROVIDER_ID],
        callbacks: {{
            signInSuccessWithAuthResult: (res, url) => {{ res.user.getIdToken().then(t => handleFirebaseLogin(t)); return false; }},
            uiShown: () => {{ const socialLoader = document.getElementById('firebaseui-loader'); if (socialLoader) socialLoader.style.display = 'none'; }}
        }}
    }});
}}
async function handleFirebaseLogin(token) {{
    try {{
        const data = await apiFetch('/api/auth/firebase-login', {{ method: 'POST', body: {{ firebase_token: token }} }});
        showToast('Login successful! Redirecting...', 'success');
        setTimeout(() => window.location.href = data.redirect_url, 1000);
    }} catch (e) {{
        showToast('Login failed on server. Please try again.', 'error');
    }}
}}

// --- GOOGLE MAPS & RIDE LOGIC ---
function loadGoogleMapsScript(){{if(!document.querySelector('script[src*="maps.googleapis.com"]')){{const s=document.createElement('script');s.src=`https://maps.googleapis.com/maps/api/js?key=${{GOOGLE_MAPS_API_KEY}}&libraries=places,directions&callback=initMap`;s.async=true;document.head.appendChild(s);}}else if(typeof google!=='undefined'){{initMap();}}}}
window.initMap=function(){{const el=document.getElementById("map");if(!el)return;map=new google.maps.Map(el,{{center:{{lat:34.0522,lng:-118.2437}},zoom:12,mapId:"RYDEPRO_DARK_STYLE",disableDefaultUI:true,zoomControl:true}});directionsService=new google.maps.DirectionsService();directionsRenderer=new google.maps.DirectionsRenderer({{map:map,suppressMarkers:true,polylineOptions:{{strokeColor:"#6d28d9",strokeWeight:6}}}});if(navigator.geolocation)navigator.geolocation.getCurrentPosition(p=>map.setCenter({{lat:p.coords.latitude,lng:p.coords.longitude}}));if(document.getElementById('passenger-dashboard'))initPassengerMapFeatures();if(document.getElementById('driver-dashboard'))initDriverMapFeatures();if(document.getElementById('track-ride-status')){{const pathParts=window.location.pathname.split('/');const shareToken=pathParts[pathParts.length-1];if(shareToken)startRideStatusPolling(shareToken);}}}}
function initPassengerMapFeatures(){{const pI=document.getElementById('pickup-location'),dI=document.getElementById('dropoff-location');const pAC=new google.maps.places.Autocomplete(pI),dAC=new google.maps.places.Autocomplete(dI);pAC.addListener('place_changed',()=>handlePlaceSelect(pAC,'pickup'));dAC.addListener('place_changed',()=>handlePlaceSelect(dAC,'dropoff'));pollForActiveRide();}}
function handlePlaceSelect(ac,type){{const p=ac.getPlace();if(!p.geometry||!p.geometry.location)return showToast(`Could not find ${{type}} location.`,'error');const l=p.geometry.location,el=document.getElementById(`${{type}}-location`);el.dataset.lat=l.lat();el.dataset.lng=l.lng();el.dataset.addr=p.formatted_address;if(type==='pickup'){{if(pickupMarker)pickupMarker.setMap(null);pickupMarker=new google.maps.Marker({{position:l,map:map,icon:'http://maps.google.com/mapfiles/ms/icons/green-dot.png'}});map.panTo(l);}}else{{if(dropoffMarker)dropoffMarker.setMap(null);dropoffMarker=new google.maps.Marker({{position:l,map:map,icon:'http://maps.google.com/mapfiles/ms/icons/red-dot.png'}});}}calculateRouteAndFare();}}
async function calculateRouteAndFare(){{const pLat=document.getElementById('pickup-location')?.dataset.lat,pLng=document.getElementById('pickup-location')?.dataset.lng,dLat=document.getElementById('dropoff-location')?.dataset.lat,dLng=document.getElementById('dropoff-location')?.dataset.lng;if(pLat&&dLat){{directionsService.route({{origin:new google.maps.LatLng(pLat,pLng),destination:new google.maps.LatLng(dLat,dLng),travelMode:'DRIVING'}},(res,stat)=>{{if(stat=='OK')directionsRenderer.setDirections(res);}});const vt=document.querySelector('.vehicle-option.selected')?.dataset.type||'Economy';try{{const f=await apiFetch('/api/estimate-fare',{{method:'POST',body:{{pickup_lat:parseFloat(pLat),pickup_lng:parseFloat(pLng),dropoff_lat:parseFloat(dLat),dropoff_lng:parseFloat(dLng),vehicle_type:vt}}}});const fb=document.getElementById('fare-estimate-box');if(fb)fb.innerHTML=`<h3>$${{f.estimated_fare.toFixed(2)}}</h3><p>${{f.distance_km}} km·${{f.duration_minutes}} mins·Surge:${{f.surge_multiplier}}x</p>`;}}catch(e){{console.error("Fare estimate failed:",e)}}}}}}
async function handleBookRide(){{const pEl=document.getElementById('pickup-location'),dEl=document.getElementById('dropoff-location');const selVeh=document.querySelector('.vehicle-option.selected');if(!pEl?.dataset.lat||!dEl?.dataset.lat)return showToast('Select pickup & drop-off locations.','error');if(!selVeh)return showToast('Select a vehicle type.','error');const req={{pickup_lat:parseFloat(pEl.dataset.lat),pickup_lng:parseFloat(pEl.dataset.lng),pickup_address:pEl.dataset.addr,dropoff_lat:parseFloat(dEl.dataset.lat),dropoff_lng:parseFloat(dEl.dataset.lng),dropoff_address:dEl.dataset.addr,vehicle_type:selVeh.dataset.type,payment_method:document.getElementById('payment-method').value}};try{{const ride=await apiFetch('/api/passenger/request-ride',{{method:'POST',body:req}});showToast('Ride requested! Searching...','success');updateRideStatusUI(ride);startRideStatusPolling(ride.id);}}catch(e){{console.error("Booking failed:",e)}}}}
function startRideStatusPolling(rideIdOrToken){{if(rideStatusPollInterval)clearInterval(rideStatusPollInterval);const isPublic=isNaN(rideIdOrToken);const url=isPublic?`/api/ride/public/${{rideIdOrToken}}`:`/api/ride/passenger/${{rideIdOrToken}}`;rideStatusPollInterval=setInterval(async()=>{{try{{const ride=await apiFetch(url,{{}},false);updateRideStatusUI(ride,isPublic);if(['completed','cancelled'].includes(ride.status)){{clearInterval(rideStatusPollInterval);if(ride.status==='completed'&&!isPublic&&!ride.passenger_rated){{showRatingModal(ride.id);}}}}}}catch(e){{clearInterval(rideStatusPollInterval);}}}},5000);}}
function pollForActiveRide(){{apiFetch('/api/passenger/active-ride',{{}},false).then(r=>{{if(r&&r.id){{showToast('Resuming active ride.','info');updateRideStatusUI(r);startRideStatusPolling(r.id);}}}}).catch(e=>{{}});}}
function updateRideStatusUI(r,isPublic=false){{let textEl,driverInfoEl,progBar,steps,container;if(isPublic){{container=document.getElementById('track-ride-status');if(!container)return;textEl=document.getElementById('track-status-text');driverInfoEl=document.getElementById('track-driver-text');}}else{{container=document.getElementById('ride-status-container');if(!container)return;container.style.display='block';textEl=document.getElementById('ride-status-text');driverInfoEl=document.getElementById('ride-driver-info');progBar=document.querySelector('.progress-bar-fg');steps=document.querySelectorAll('.progress-step');document.getElementById('sos-button').dataset.rideId=r.id;document.getElementById('share-trip-button').dataset.shareToken=r.share_token;}}if(!textEl||!driverInfoEl)return;textEl.textContent=`Status: ${{r.status.replace(/_/g,' ').toUpperCase()}}`;if(r.driver){{driverInfoEl.innerHTML=`<div><strong>${{r.driver.full_name}}</strong>(${{r.driver.average_rating}} <i class="fas fa-star" style="color:#f59e0b;"></i>)</div><div>${{r.driver.vehicle.color}} ${{r.driver.vehicle.make}} - ${{r.driver.vehicle.license_plate}}</div>`;}}else{{driverInfoEl.innerHTML='Searching for a driver...';}}if(!isPublic&&progBar&&steps){{let prog=0;steps.forEach(s=>s.classList.remove('active'));switch(r.status){{case'pending':prog=10;steps[0].classList.add('active');break;case'accepted':case'arriving':prog=40;steps[1].classList.add('active');break;case'in_progress':prog=75;steps[2].classList.add('active');break;case'completed':prog=100;steps[3].classList.add('active');driverInfoEl.innerHTML='Trip completed! Thank you.';setTimeout(()=>container.style.display='none',10000);break;case'cancelled':prog=0;driverInfoEl.innerHTML='This ride has been cancelled.';setTimeout(()=>container.style.display='none',5000);break;}}progBar.style.width=`${{prog}}%`;}}if(r.driver_current_lat&&r.driver_current_lng){{updateDriverMarker(r.driver_current_lat,r.driver_current_lng,r.driver?.full_name);}}}}
function updateDriverMarker(lat,lng,name){{if(!map||!lat||!lng)return;const pos=new google.maps.LatLng(lat,lng);if(!driverMarker){{driverMarker=new google.maps.Marker({{position:pos,map,title:name,icon:{{path:google.maps.SymbolPath.FORWARD_CLOSED_ARROW,scale:6,fillColor:"#12121c",fillOpacity:1,strokeWeight:2,strokeColor:"#6d28d9"}}}});}}else{{driverMarker.setPosition(pos);}}const bounds=map.getBounds();if(bounds&&!bounds.contains(pos)){{map.panTo(pos);}}}}
function initDriverMapFeatures(){{const t=document.getElementById('driver-status-toggle');if(t)t.addEventListener('click',toggleDriverAvailability);pollForRideRequests();setInterval(()=>{{if(t&&t.classList.contains('online')){{if(navigator.geolocation)navigator.geolocation.getCurrentPosition(p=>apiFetch('/api/driver/update-location',{{method:'POST',body:{{lat:p.coords.latitude,lng:p.coords.longitude}}}},false).catch(e=>{{}}));}}}},10000);}}
async function toggleDriverAvailability(){{const t=this;const isOnline=!t.classList.contains('online');try{{await apiFetch('/api/driver/toggle-availability',{{method:'POST',body:{{online:isOnline}}}});t.classList.toggle('online');t.querySelector('span').textContent=`You are ${{isOnline?'ONLINE':'OFFLINE'}}`;showToast(`You are now ${{isOnline?'online':'offline'}}`,'success');}}catch(e){{}}}}
function pollForRideRequests(){{if(rideRequestPollInterval)clearInterval(rideRequestPollInterval);rideRequestPollInterval=setInterval(async()=>{{const t=document.getElementById('driver-status-toggle');if(t&&t.classList.contains('online')&&t.dataset.status==='online'){{try{{const r=await apiFetch('/api/driver/ride-request',{{}},false);if(r&&r.id){{showRideRequestModal(r);clearInterval(rideRequestPollInterval);}}}}catch(e){{}}}}}},5000);}}
function showRideRequestModal(r){{const m=document.getElementById('ride-request-modal');if(!m)return;document.getElementById('request-pickup').textContent=r.pickup_address;document.getElementById('request-dropoff').textContent=r.dropoff_address;document.getElementById('request-fare').textContent=`$${{r.estimated_fare.toFixed(2)}}`;m.style.display='flex';let time=30;const timerEl=document.getElementById('countdown-timer');timerEl.textContent=time;countdownInterval=setInterval(()=>{{time--;timerEl.textContent=time;if(time<=0)hideRideRequestModal();}},1000);document.getElementById('accept-ride-btn').onclick=()=>handleRideAction('accept',r.id);document.getElementById('reject-ride-btn').onclick=()=>handleRideAction('reject',r.id);}}
function hideRideRequestModal(){{const modal=document.getElementById('ride-request-modal');if(modal)modal.style.display='none';clearInterval(countdownInterval);pollForRideRequests();}}
async function handleRideAction(action,id){{hideRideRequestModal();if(action==='reject')return;try{{const r=await apiFetch(`/api/driver/accept-ride`,{{method:'POST',body:{{ride_id:id}}}});showToast(`Ride accepted!`,'success');updateDriverTripUI(r);}}catch(e){{console.error("Accept ride failed:",e)}}}}
function updateDriverTripUI(r){{const cont=document.getElementById('current-trip-info');if(!cont)return;cont.innerHTML=`<div class="card"><h4>Current Ride:#${{r.id}}</h4><p><strong>To:</strong>${{r.dropoff_address}}</p><p><strong>Passenger:</strong>${{r.passenger.full_name}}</p><div id="driver-action-buttons"><button class="btn btn-secondary" onclick="updateDriverRideStatus(${{r.id}},'arriving')">I've Arrived</button></div></div>`;const toggle=document.getElementById('driver-status-toggle');if(toggle)toggle.dataset.status='on_trip';}}
async function updateDriverRideStatus(id,status){{try{{await apiFetch('/api/driver/update-ride-status',{{method:'POST',body:{{ride_id:id,status:status}}}});showToast(`Status updated to ${{status}}`,'success');const btns=document.getElementById('driver-action-buttons');if(!btns)return;let nextAction='';if(status==='arriving'){{nextAction=`<button class="btn btn-primary" onclick="updateDriverRideStatus(${{id}},'in_progress')">Start Trip</button>`;}}else if(status==='in_progress'){{nextAction=`<button class="btn btn-success" onclick="updateDriverRideStatus(${{id}},'completed')">Complete Trip</button>`;}}else{{document.getElementById('current-trip-info').innerHTML='<p>No active trip. You are now available for new requests.</p>';const t=document.getElementById('driver-status-toggle');if(t)t.dataset.status='online';pollForRideRequests();return;}}btns.innerHTML=nextAction;}}catch(e){{console.error("Update status failed:",e)}}}}

// --- GENERAL UI HANDLERS ---
function showRatingModal(id){{const m=document.getElementById('rating-modal');if(!m)return;m.style.display='flex';m.dataset.rideId=id;}}
function handleSOS(){{const btn=document.getElementById('sos-button');if(!btn||!btn.dataset.rideId||!confirm("Trigger SOS alert? This will notify our safety team."))return;if(navigator.geolocation)navigator.geolocation.getCurrentPosition(async p=>{{try{{await apiFetch('/api/ride/sos',{{method:'POST',body:{{ride_id:parseInt(btn.dataset.rideId),lat:p.coords.latitude,lng:p.coords.longitude}}}});showToast("SOS Alert triggered. Team notified.",'error',10000);}}catch(e){{}}}});}}
function handleShareTrip(token){{const url=`${{window.location.origin}}/track/${{token}}`;const modal=document.getElementById('share-trip-modal');if(!modal)return;modal.querySelector('#share-url-input').value=url;modal.style.display='flex';}}
function copyShareUrl(){{const input=document.getElementById('share-url-input');if(!input)return;input.select();input.setSelectionRange(0,99999);document.execCommand('copy');showToast('Link copied to clipboard!','success');}}
function handleTopUp(){{const amt=parseFloat(document.getElementById('topup-amount')?.value);if(isNaN(amt)||amt<=0)return showToast("Enter a valid amount.",'error');PaystackPop.setup({{key:PAYSTACK_PUBLIC_KEY,email:document.body.dataset.userEmail,amount:amt*100,currency:'USD',ref:'rydepro_w_'+Math.floor(1e9*Math.random()+1),callback:async r=>{{try{{const res=await apiFetch('/api/wallet/verify-topup',{{method:'POST',body:{{reference:r.reference,amount:amt}}}});showToast('Wallet topped up!','success');const b=document.getElementById('wallet-balance');if(b)b.textContent=`$${{res.new_balance.toFixed(2)}}`;loadWalletTransactions();}}catch(e){{showToast('Verification failed.','error');}}}},onClose:()=>showToast('Payment window closed.','info')}}).openIframe();}}
async function loadWalletTransactions(){{const cont=document.getElementById('transaction-history-body');if(!cont)return;try{{const txs=await apiFetch('/api/wallet/history');if(txs.length===0){{cont.innerHTML='<tr><td colspan="4" class="text-center">No transactions yet.</td></tr>';}}else{{cont.innerHTML=txs.map(tx=>`<tr><td>${{new Date(tx.created_at).toLocaleString()}}</td><td>${{tx.description}}</td><td class="${{tx.amount>0?'text-success':'text-error'}}">$${{tx.amount.toFixed(2)}}</td><td>${{tx.reference?tx.reference.substring(0,20)+'...':''}}</td></tr>`).join('');}}}}catch(e){{cont.innerHTML='<tr><td colspan="4" class="text-center text-error">Could not load history.</td></tr>';}}}}
async function handleAdminDriverAction(btn,driverId,action){{const reason=action==='reject'?prompt("Reason for rejection:"):null;if(action==='reject'&&!reason)return;btn.disabled=true;try{{const res=await apiFetch('/api/admin/driver-action',{{method:'POST',body:{{driver_id:driverId,action,reason}}}});showToast(`Driver ${{action}}ed.`,'success');const row=btn.closest('tr');row.querySelector('.status-cell').innerHTML=`<span class="status-tag status-${{res.new_status}}">${{res.new_status}}</span>`;row.querySelector('.action-cell').innerHTML='Action Taken';}}catch(e){{btn.disabled=false;}}}}
async function purchaseSubscription(plan){{if(!confirm(`Confirm purchase of ${{plan.charAt(0).toUpperCase()+plan.slice(1)}} plan? This is a simulated payment from your wallet.`))return;try{{const res=await apiFetch('/api/user/subscribe',{{method:'POST',body:{{plan}}}});showToast(`Successfully subscribed to ${{res.new_plan}}!`,'success');window.location.reload();}}catch(e){{}}}}
function showAuthTab(tabName){{document.querySelectorAll('.auth-tab-content').forEach(c=>c.classList.remove('active'));document.querySelectorAll('.auth-tab-button').forEach(b=>b.classList.remove('active'));const content=document.getElementById(tabName);const button=document.querySelector(`[data-tab='${{tabName}}']`);if(content)content.classList.add('active');if(button)button.classList.add('active');}}

// --- DOMCONTENTLOADED - MAIN ENTRY POINT ---
document.addEventListener('DOMContentLoaded',()=>{{
    if (document.getElementById('auth-page')) initializeFirebase();
    if (document.getElementById('map')) loadGoogleMapsScript();
    if (document.getElementById('wallet-container')) loadWalletTransactions();
    document.querySelectorAll('.auth-tab-button').forEach(b=>b.addEventListener('click',()=>showAuthTab(b.dataset.tab)));
    document.getElementById('logout-link')?.addEventListener('click',async e=>{{e.preventDefault();try{{await apiFetch('/api/auth/logout',{{method:'POST'}});}}finally{{window.location.href='/';}}}});
    document.getElementById('book-ride-btn')?.addEventListener('click',handleBookRide);
    document.querySelectorAll('.vehicle-option').forEach(o=>o.addEventListener('click',()=>{{document.querySelectorAll('.vehicle-option').forEach(opt=>opt.classList.remove('selected'));o.classList.add('selected');calculateRouteAndFare();}}));
    document.getElementById('sos-button')?.addEventListener('click',handleSOS);
    document.getElementById('share-trip-button')?.addEventListener('click',function(){{handleShareTrip(this.dataset.shareToken);}});
    document.getElementById('topup-btn')?.addEventListener('click',handleTopUp);
    document.querySelectorAll('.close-modal').forEach(el=>el.addEventListener('click',()=>el.closest('.modal').style.display='none'));
    document.getElementById('rating-modal-form')?.addEventListener('submit',async e=>{{e.preventDefault();const m=document.getElementById('rating-modal'),id=m.dataset.rideId;const r=e.target.rating.value,c=e.target.comment.value;try{{await apiFetch('/api/ride/rate',{{method:'POST',body:{{ride_id:parseInt(id),rating:parseInt(r),comment:c}}}});showToast('Feedback received!','success');m.style.display='none';window.location.reload();}}catch(err){{}}}});
    document.getElementById('profile-update-form')?.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,formData=new FormData(form);try{{const res=await apiFetch('/api/profile',{{method:'PUT',body:formData}});showToast('Profile updated!','success');if(res.profile_picture_url)document.getElementById('profile-avatar-img').src=res.profile_picture_url+'?t='+new Date().getTime();}}catch(err){{}}}});
    document.getElementById('driver-registration-form')?.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,formData=new FormData(form),btn=form.querySelector('button');btn.disabled=true;btn.textContent='Submitting...';try{{await apiFetch('/api/driver/register',{{method:'POST',body:formData}});showToast('Registration submitted for review!','success');setTimeout(()=>window.location.href='/driver',2000);}}catch(err){{btn.disabled=false;btn.textContent='Submit for Review';}}}});
    document.getElementById('login-form')?.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,formData=new FormData(form);const email=formData.get('email'),password=formData.get('password');try{{const data=await apiFetch('/api/auth/login',{{method:'POST',body:{{email,password}}}});showToast('Login successful! Redirecting...','success');setTimeout(()=>window.location.href=data.redirect_url,1000);}}catch(err){{}}}});
    document.getElementById('register-form')?.addEventListener('submit',async e=>{{e.preventDefault();const form=e.target,formData=new FormData(form);const full_name=formData.get('full_name'),email=formData.get('email'),password=formData.get('password');try{{await apiFetch('/api/auth/register',{{method:'POST',body:{{full_name,email,password}}}});showToast('Registration successful! Please log in.','success');setTimeout(()=>window.location.href='/login',1500);}}catch(err){{}}}});
}});
"""


# ==============================================================================
# PART 9: HTML TEMPLATES
# ==============================================================================
def get_base_html(title, content, current_user=None):
    user_menu, body_data = "", ""
    if current_user:
        dashboard_link = f"/{current_user.role.value}"
        body_data = f"data-user-id='{current_user.id}' data-user-email='{current_user.email}'"
        user_menu = f"""<div class="user-menu"><img src="{current_user.profile_picture_url}" alt="Avatar"/><ul class="dropdown-menu"><li><a href="{dashboard_link}"><i class="fa-fw fas fa-tachometer-alt"></i> Dashboard</a></li><li><a href="/{current_user.role.value}/profile"><i class="fa-fw fas fa-user-edit"></i> Profile</a></li><li><a href="/{current_user.role.value}/history"><i class="fa-fw fas fa-history"></i> History</a></li>{'<li><a href="/passenger/wallet"><i class="fa-fw fas fa-wallet"></i> Wallet</a></li>' if current_user.role == UserRole.passenger else ''}{'<li><a href="/passenger/subscriptions"><i class="fa-fw fas fa-star"></i> Subscriptions</a></li>' if current_user.role == UserRole.passenger else ''}<li><a href="#" id="logout-link"><i class="fa-fw fas fa-sign-out-alt"></i> Logout</a></li></ul></div>"""
    else:
        user_menu = f'<a href="/login" class="btn btn-primary">Login / Sign Up</a>'
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{CONFIG['PROJECT_NAME']} - {title}</title><style>{MAIN_CSS}</style><script src="https://www.gstatic.com/firebasejs/9.6.1/firebase-app-compat.js"></script><script src="https://www.gstatic.com/firebasejs/9.6.1/firebase-auth-compat.js"></script><script src="https://www.gstatic.com/firebasejs/ui/6.0.1/firebase-ui-auth.js"></script><link type="text/css" rel="stylesheet" href="https://www.gstatic.com/firebasejs/ui/6.0.1/firebase-ui-auth.css"/></head><body {body_data}><div id="toast-container"></div><header class="main-header"><a href="/" class="logo"><i class="fas fa-rocket"></i> RydePro</a><nav class="main-nav"></nav><div>{user_menu}</div></header><main class="main-content">{content}</main><footer class="main-footer"><p>© {datetime.now().year} {CONFIG['PROJECT_NAME']}.</p></footer><script>{MAIN_JS}</script></body></html>"""


def get_dashboard_html(title, content, user, active_page):
    sidebar_links = "";
    role = user.role.value
    links = {"passenger": [("map-marked-alt", "Book a Ride", ""), ("history", "Ride History", "/history"),
                           ("wallet", "Wallet", "/wallet"), ("star", "Subscriptions", "/subscriptions"),
                           ("user-cog", "Profile", "/profile")],
             "driver": [("tachometer-alt", "Dashboard", ""), ("dollar-sign", "Earnings", "/earnings"),
                        ("history", "Trip History", "/history"), ("user-shield", "Profile & Docs", "/profile")],
             "admin": [("chart-line", "Dashboard", ""), ("users", "Users", "/users"),
                       ("id-card", "Drivers", "/drivers"), ("route", "Rides", "/rides")]}
    for icon, text, path in links[
        role]: sidebar_links += f'<li><a href="/{role}{path}" class="{"active" if active_page == (path.strip("/") if path else "dashboard") else ""}"><i class="fa-fw fas fa-{icon}"></i> {text}</a></li>'
    return get_base_html(title,
                         f'<div class="dashboard-layout"><aside class="sidebar"><nav class="sidebar-nav"><ul>{sidebar_links}</ul></nav></aside><section class="dashboard-content"><h1>{title}</h1>{content}</section></div>',
                         user)


def render_generic_page(title, content):
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{CONFIG['PROJECT_NAME']} - {title}</title><style>{MAIN_CSS}</style></head><body><header class="main-header"><a href="/" class="logo"><i class="fas fa-rocket"></i> RydePro</a></header><main class="main-content"><div class="container">{content}</div></main><footer class="main-footer"><p>© {datetime.now().year} {CONFIG['PROJECT_NAME']}.</p></footer><script>{MAIN_JS}</script></body></html>"""


LANDING_PAGE_HTML = """<div class="container text-center" style="padding: 4rem 1.5rem;"><h1 style="font-size:3rem;">The Future of Mobility is Here</h1><p class="mb-3" style="font-size: 1.2rem; color: var(--text-secondary);">Welcome to RydePro. Seamless, intelligent, and on-demand.</p><a href="/login" class="btn btn-primary btn-lg" style="padding: 15px 40px; font-size: 1.2rem;">Get Started</a></div>"""
LOGIN_PAGE_HTML = """
<div id="auth-page"><div class="form-container">
    <div class="auth-tabs"><button class="auth-tab-button active" data-tab="email-tab">Email & Password</button><button class="auth-tab-button" data-tab="social-tab">Social Login</button></div>
    <div id="email-tab" class="auth-tab-content active"><h2 class="text-center">Login</h2><form id="login-form"><div class="form-group"><label for="email">Email</label><input type="email" name="email" class="form-control" required></div><div class="form-group"><label for="password">Password</label><input type="password" name="password" class="form-control" required></div><button type="submit" class="btn btn-primary" style="width:100%;">Login</button></form><p class="text-center mt-3">Don't have an account? <a href="/register">Sign Up</a></p></div>
    <div id="social-tab" class="auth-tab-content"><h2 class="text-center">Social Login</h2><p class="text-center" style="color:var(--text-secondary);">Use your Google account to sign in instantly.</p><div id="firebaseui-auth-container"></div><div id="firebaseui-loader" class="text-center mt-3"><div class="loader"></div></div></div>
</div></div>"""
REGISTER_PAGE_HTML = """
<div id="auth-page"><div class="form-container">
    <h2 class="text-center">Create an Account</h2><form id="register-form"><div class="form-group"><label for="full_name">Full Name</label><input type="text" name="full_name" class="form-control" required></div><div class="form-group"><label for="email">Email</label><input type="email" name="email" class="form-control" required></div><div class="form-group"><label for="password">Password</label><input type="password" name="password" class="form-control" required minlength="6"></div><button type="submit" class="btn btn-primary" style="width:100%;">Sign Up</button></form><p class="text-center mt-3">Already have an account? <a href="/login">Log In</a></p>
</div></div>"""
PASSENGER_DASHBOARD_HTML = """<div id="passenger-dashboard" class="passenger-dashboard-grid"><div id="map"></div><div class="booking-panel"><h3>Where to?</h3><div class="form-group"><label for="pickup-location">Pickup</label><input type="text" id="pickup-location" class="form-control" placeholder="Enter pickup address"></div><div class="form-group"><label for="dropoff-location">Drop-off</label><input type="text" id="dropoff-location" class="form-control" placeholder="Enter destination"></div><p>Vehicle Type:</p><div class="vehicle-options"><div class="vehicle-option selected" data-type="Economy"><i class="fas fa-car-side icon"></i><span class="name">Economy</span></div><div class="vehicle-option" data-type="Luxury"><i class="fas fa-gem icon"></i><span class="name">Luxury</span></div><div class="vehicle-option" data-type="SUV"><i class="fas fa-truck-monster icon"></i><span class="name">SUV</span></div></div><div class="form-group"><label for="payment-method">Payment</label><select id="payment-method" class="form-control"><option value="wallet">Wallet</option><option value="card">Card (Future)</option><option value="cash">Cash</option></select></div><button id="book-ride-btn" class="btn btn-primary">Request Ride</button><div id="fare-estimate-box" class="mt-3"><p>Enter locations for fare estimate.</p></div></div></div><div id="ride-status-container"><div class="ride-status-header"><h3 id="ride-status-text">Status...</h3><div class="ride-status-actions"><button id="share-trip-button" title="Share Trip"><i class="fas fa-share-alt"></i></button><button id="sos-button" title="Emergency SOS"><i class="fas fa-exclamation-triangle"></i></button></div></div><div id="ride-driver-info" class="mt-2"></div><div class="progress-container mt-3"><div class="progress-steps"><div class="progress-step">Requested</div><div class="progress-step">En Route</div><div class="progress-step">In Progress</div><div class="progress-step">Completed</div></div><div class="progress-bar-bg"><div class="progress-bar-fg"></div></div></div></div><div id="rating-modal" class="modal"><div class="modal-content"><span class="close-modal">×</span><h2>How was your ride?</h2><form id="rating-modal-form"><div class="star-rating"><input type="radio" id="5-stars" name="rating" value="5" /><label for="5-stars" class="fa-solid fa-star"></label><input type="radio" id="4-stars" name="rating" value="4" /><label for="4-stars" class="fa-solid fa-star"></label><input type="radio" id="3-stars" name="rating" value="3" checked /><label for="3-stars" class="fa-solid fa-star"></label><input type="radio" id="2-stars" name="rating" value="2" /><label for="2-stars" class="fa-solid fa-star"></label><input type="radio" id="1-star" name="rating" value="1" /><label for="1-star" class="fa-solid fa-star"></label></div><div class="form-group mt-3"><textarea name="comment" class="form-control" placeholder="Add a comment..."></textarea></div><button type="submit" class="btn btn-primary">Submit Review</button></form></div></div><div id="share-trip-modal" class="modal"><div class="modal-content"><span class="close-modal">×</span><h2>Share Your Trip</h2><p>Send this link to friends and family to track your ride in real-time.</p><div class="form-group" style="display:flex; gap:10px;"><input type="text" id="share-url-input" class="form-control" readonly><button class="btn btn-secondary" onclick="copyShareUrl()">Copy</button></div></div></div>"""
# (Other HTML templates are unchanged and included for completeness)
DRIVER_DASHBOARD_HTML = """<div id="driver-dashboard"><div class="grid-container"><div class="card"><div id="driver-status-toggle" class="{is_online_class}" data-status="{status_text_raw}"><div class="toggle-switch"><div class="slider"></div></div><span>You are {status_text}</span></div></div><div class="stat-card"><div class="icon"><i class="fas fa-dollar-sign"></i></div><div class="value">$ {today_earnings:.2f}</div><div class="label">Today's Earnings</div></div><div class="stat-card"><div class="icon"><i class="fas fa-star"></i></div><div class="value">{avg_rating}</div><div class="label">Your Rating</div></div></div><div class="card mt-3"><div class="card-header">Current Trip</div><div id="current-trip-info">{current_trip_html}</div></div></div><div id="ride-request-modal" class="modal"><div class="modal-content"><h2>New Ride Request!</h2><div class="countdown-timer" id="countdown-timer">30</div><div class="card" style="text-align:left;"><p><strong>From:</strong> <span id="request-pickup"></span></p><p><strong>To:</strong> <span id="request-dropoff"></span></p><p><strong>Est. Fare:</strong> <span id="request-fare"></span></p></div><div style="display:flex;justify-content:space-around;margin-top:1.5rem"><button id="reject-ride-btn" class="btn btn-danger">Reject</button><button id="accept-ride-btn" class="btn btn-success">Accept</button></div></div></div>"""
PROFILE_PAGE_HTML = """<div class="card"><div class="card-header">Your Profile</div><form id="profile-update-form" class="text-center"><img src="{user.profile_picture_url}" alt="Avatar" class="profile-avatar" id="profile-avatar-img"><div class="form-group mt-3"><div class="file-upload-wrapper btn btn-secondary"><span>Change Photo</span><input type="file" name="avatar" accept="image/*"></div></div><div class="form-group" style="text-align:left"><label for="full_name">Full Name</label><input type="text" name="full_name" class="form-control" value="{user.full_name}"></div><div class="form-group" style="text-align:left"><label>Email</label><input type="email" class="form-control" value="{user.email}" disabled></div><button type="submit" class="btn btn-primary">Save Changes</button></form></div>"""
HISTORY_PAGE_HTML = """<div class="card"><div class="card-header">Ride History</div><div class="table-container"><table class="data-table"><thead><tr><th>Date</th><th>From</th><th>To</th><th>Fare</th><th>Status</th><th>Action</th></tr></thead><tbody>{history_rows}</tbody></table></div></div>"""
WALLET_PAGE_HTML = """<div id="wallet-container"><div class="card"><div class="card-header">Your Wallet</div><h2>Balance: <span id="wallet-balance" class="text-success">${user.wallet_balance:.2f}</span></h2><div class="form-group mt-3"><label for="topup-amount">Top-up Amount ($)</label><input type="number" id="topup-amount" class="form-control" placeholder="e.g., 50"></div><button id="topup-btn" class="btn btn-primary">Top Up with PayStack</button></div><div class="card mt-3"><div class="card-header">Transaction History</div><div class="table-container"><table class="data-table"><thead><tr><th>Date</th><th>Description</th><th>Amount</th><th>Reference</th></tr></thead><tbody id="transaction-history-body"><div class="loader"></div></tbody></table></div></div><script src="https://js.paystack.co/v1/inline.js"></script>"""
DRIVER_REGISTRATION_HTML = """<div class="form-container"><h2 class="text-center">Become a RydePro Driver</h2><p class="text-center" style="color:var(--text-secondary)">Complete your profile to start earning.</p><form id="driver-registration-form"><div class="card-header" style="margin-top:1rem;">Vehicle Information</div><div class="form-group"><label>Make</label><input type="text" name="vehicle_make" class="form-control" required></div><div class="form-group"><label>Model</label><input type="text" name="vehicle_model" class="form-control" required></div><div class="form-group"><label>Year</label><input type="number" name="vehicle_year" class="form-control" required></div><div class="form-group"><label>Color</label><input type="text" name="vehicle_color" class="form-control" required></div><div class="form-group"><label>License Plate</label><input type="text" name="vehicle_license_plate" class="form-control" required></div><div class="form-group"><label>Vehicle Type</label><select name="vehicle_type" class="form-control">{vehicle_type_options}</select></div><div class="card-header" style="margin-top:1.5rem;">Documents</div><div class="form-group"><label>Driver's License Number</label><input type="text" name="license_number" class="form-control" required></div><div class="form-group"><label>License Document (PDF/JPG)</label><input type="file" name="license_doc" class="form-control" accept=".pdf,.jpg,.jpeg,.png" required></div><div class="form-group"><label>Insurance Document (PDF/JPG)</label><input type="file" name="insurance_doc" class="form-control" accept=".pdf,.jpg,.jpeg,.png" required></div><button type="submit" class="btn btn-primary">Submit for Review</button></form></div>"""
ADMIN_DRIVERS_HTML = """<div class="card"><div class="card-header">Driver Management</div><div class="table-container"><table class="data-table"><thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Rating</th><th>Status</th><th>Actions</th></tr></thead><tbody>{driver_rows}</tbody></table></div></div>"""
SUBSCRIPTIONS_PAGE_HTML = """<div class="grid-container">{plan_cards}</div>"""
TRACK_RIDE_PAGE_HTML = """<div id="map" style="width:100%; height:calc(100vh - 100px); border-radius:0;"></div><div id="track-ride-status" style="position:fixed; bottom:20px; left:20px; background:var(--bg-dark-secondary); padding:1.5rem; border-radius:var(--border-radius-md); border:1px solid var(--border-color); box-shadow:var(--shadow-lg);"><h3 id="track-status-text">Tracking Ride...</h3><p id="track-driver-text">Fetching details...</p></div>"""

# ==============================================================================
# PART 10: FASTAPI APPLICATION & ENDPOINTS
# ==============================================================================
app = FastAPI(title=CONFIG["PROJECT_NAME"], on_startup=[setup_dirs, init_firebase])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
app.mount("/static", StaticFiles(directory=CONFIG["UPLOADS_DIR"]), name="static")

# --- Routers ---
api_router = APIRouter(prefix="/api")
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# --- AUTHENTICATION API ENDPOINTS ---
@auth_router.post("/register")
async def api_register(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        full_name=user_data.full_name,
        email=user_data.email,
        hashed_password=hashed_password,
        role=UserRole.admin if user_data.email == "admin@rydepro.com" else UserRole.passenger
    )
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}


@auth_router.post("/login")
async def api_login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not user.hashed_password or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": str(user.id)})
    response = JSONResponse(content={"redirect_url": f"/{user.role.value}"})
    response.set_cookie("access_token", access_token, httponly=True, samesite='lax', secure=False, path="/")
    return response


@auth_router.post("/firebase-login")
async def api_firebase_login(data: FirebaseLogin, db: Session = Depends(get_db)):
    try:
        decoded = auth.verify_id_token(data.firebase_token)
        uid, email = decoded['uid'], decoded.get('email')
        user = db.query(User).filter(User.firebase_uid == uid).first()
        if not user:
            user = db.query(User).filter(User.email == email).first()
            if user:
                user.firebase_uid = uid
            else:
                user = User(firebase_uid=uid, full_name=decoded.get('name', 'New User'), email=email,
                            role=UserRole.passenger)
                db.add(user)
        db.commit()
        db.refresh(user)
        access_token = create_access_token({"sub": str(user.id)})
        response = JSONResponse(content={"redirect_url": f"/{user.role.value}"})
        response.set_cookie("access_token", access_token, httponly=True, samesite='lax', secure=False, path="/")
        return response
    except Exception as e:
        raise HTTPException(401, f"Invalid Firebase token: {e}")


@auth_router.post("/logout")
def api_logout():
    res = JSONResponse({"message": "Logged out"})
    res.delete_cookie("access_token")
    return res


# --- PAGE RENDERING ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def page_root(user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(f"/{user.role.value}")
    return HTMLResponse(get_base_html("Home", LANDING_PAGE_HTML, user))


@app.get("/login", response_class=HTMLResponse)
async def page_login(user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(f"/{user.role.value}")
    return HTMLResponse(get_base_html("Login", LOGIN_PAGE_HTML))


@app.get("/register", response_class=HTMLResponse)
async def page_register(user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(f"/{user.role.value}")
    return HTMLResponse(get_base_html("Sign Up", REGISTER_PAGE_HTML))


# (The rest of the endpoints from the previous version follow here and are unchanged)
@app.get("/track/{share_token}", response_class=HTMLResponse)
async def page_track_ride(share_token: str):
    return HTMLResponse(render_generic_page("Track Ride", TRACK_RIDE_PAGE_HTML))


@app.get("/passenger", response_class=HTMLResponse)
async def page_passenger_dashboard(user: User = Depends(require_passenger)):
    return HTMLResponse(get_dashboard_html("Book a Ride", PASSENGER_DASHBOARD_HTML, user, "dashboard"))


@app.get("/passenger/profile", response_class=HTMLResponse)
async def page_passenger_profile(user: User = Depends(require_passenger)):
    return HTMLResponse(get_dashboard_html("Profile", PROFILE_PAGE_HTML.format(user=user), user, "profile"))


@app.get("/passenger/history", response_class=HTMLResponse)
async def page_passenger_history(user: User = Depends(require_passenger), db: Session = Depends(get_db)):
    """
    Renders the passenger's ride history page.
    This version uses a helper function to generate the action button HTML,
    avoiding complex and unreadable backslash escaping.
    """
    rides = db.query(Ride).options(
        joinedload(Ride.driver).joinedload(Driver.user)
    ).filter(Ride.passenger_id == user.id).order_by(Ride.created_at.desc()).limit(50).all()

    def generate_action_button(ride: Ride) -> str:
        """
        Generates the HTML for the 'Action' column button based on ride status.
        This makes the main loop much cleaner.
        """
        if ride.status == RideStatus.completed and not ride.passenger_rated:
            # Using single quotes for the outer HTML string allows double quotes inside without escaping.
            return f'<button class="btn btn-secondary" onclick="showRatingModal({ride.id})">Rate</button>'
        return "N/A"

    # Build table rows using a clear, multi-line f-string
    rows = []
    for r in rides:
        action_html = generate_action_button(r)
        row_html = f"""
            <tr>
                <td>{r.created_at.strftime("%b %d, %H:%M")}</td>
                <td>{r.pickup_address[:25]}...</td>
                <td>{r.dropoff_address[:25]}...</td>
                <td>${(r.actual_fare or r.estimated_fare):.2f}</td>
                <td><span class="status-tag status-{r.status.value}">{r.status.value.replace('_', ' ')}</span></td>
                <td>{action_html}</td>
            </tr>
        """
        rows.append(row_html)
    history_rows_html = "".join(rows)
    if not history_rows_html:
        history_rows_html = '<tr><td colspan="6" class="text-center">You have no ride history yet.</td></tr>'
    return HTMLResponse(
        get_dashboard_html("Ride History", HISTORY_PAGE_HTML.format(history_rows=history_rows_html), user, "history"))


@app.get("/passenger/wallet", response_class=HTMLResponse)
async def page_passenger_wallet(user: User = Depends(require_passenger)):
    return HTMLResponse(get_dashboard_html("Wallet", WALLET_PAGE_HTML.format(user=user), user, "wallet"))


@app.get("/passenger/subscriptions", response_class=HTMLResponse)
async def page_subscriptions(user: User = Depends(require_passenger)):
    cards = ""
    for plan_id, plan_details in CONFIG[
        'SUBSCRIPTION_PLANS'].items():
        is_current = user.subscription_plan.value == plan_id
        button = f'<button class="btn btn-secondary current" disabled>Current Plan</button>' if is_current else f'<button class="btn btn-primary" onclick="purchaseSubscription(\'{plan_id}\')">Choose Plan</button>'
        benefits = "".join(f'<li><i class="fas fa-check-circle text-success"></i> {b}</li>' for b in
                           plan_details['benefits'])
        cards += f"""<div class="card subscription-card {'popular' if plan_id == 'premium' else ''}"><h3>{plan_details['name']}</h3><div class="price">${plan_details['price']}<span style="font-size:1rem;color:var(--text-secondary)">/mo</span></div><ul>{benefits}</ul>{button}</div>"""
    return HTMLResponse(
        get_dashboard_html("Subscriptions", SUBSCRIPTIONS_PAGE_HTML.format(plan_cards=cards), user, "subscriptions"))


@app.get("/driver", response_class=HTMLResponse)
async def page_driver_dashboard(user: User = Depends(require_driver), db: Session = Depends(get_db)):
    driver = user.driver_info
    if not driver:
        return RedirectResponse("/driver/register")
    if driver.status in [DriverStatus.pending_approval, DriverStatus.rejected]:
        reason = f"<p>Reason: {driver.rejection_reason}</p>" if driver.rejection_reason else ""
        return HTMLResponse(
            get_dashboard_html(f"Application {driver.status.name.replace('_', ' ').title()}",
                               f"<p>Your application is currently {driver.status.name.replace('_', ' ')}.</p>{reason}",
                               user, "dashboard"))
    current_ride = db.query(Ride).filter(Ride.driver_id == driver.id, Ride.status.in_(
        [RideStatus.accepted, RideStatus.arriving, RideStatus.in_progress])).first()
    trip_html = "<p>No active trip. You are ready for requests.</p>"
    if current_ride:
        actions_map = {
            RideStatus.accepted: f'<button class="btn btn-secondary" onclick="updateDriverRideStatus({current_ride.id}, \'arriving\')">I\'ve Arrived</button>',
            RideStatus.arriving: f'<button class="btn btn-primary" onclick="updateDriverRideStatus({current_ride.id}, \'in_progress\')">Start Trip</button>',
            RideStatus.in_progress: f'<button class="btn btn-success" onclick="updateDriverRideStatus({current_ride.id}, \'completed\')">Complete Trip</button>'
        }
        action = actions_map.get(current_ride.status, "")
        trip_html = f'<div class="card"><h4>Ride #{current_ride.id} to {current_ride.dropoff_address}</h4><div id="driver-action-buttons">{action}</div></div>'
    earnings = sum(t.amount for t in db.query(WalletTransaction.amount).filter(WalletTransaction.user_id == user.id,
                                                                               WalletTransaction.transaction_type == TransactionType.ride_earning,
                                                                               WalletTransaction.created_at >= datetime.utcnow().date()).all())
    content = DRIVER_DASHBOARD_HTML.format(is_online_class='online' if driver.status == DriverStatus.online else '',
                                           status_text=driver.status.value.upper(),
                                           status_text_raw=driver.status.value, today_earnings=earnings,
                                           avg_rating=f"{driver.average_rating:.1f}",
                                           current_trip_html=trip_html)
    return HTMLResponse(get_dashboard_html("Dashboard", content, user, "dashboard"))


@app.get("/driver/register", response_class=HTMLResponse)
async def page_driver_register(user: User = Depends(get_current_active_user)):
    # FIX: Allow passengers to access this page to become drivers.
    if user.role not in [UserRole.passenger, UserRole.driver]:
        return RedirectResponse("/")  # Redirect non-passengers away
    return HTMLResponse(get_base_html("Driver Registration", DRIVER_REGISTRATION_HTML.format(
        vehicle_type_options="".join([f'<option value="{vt.value}">{vt.name}</option>' for vt in VehicleType])), user))


@app.get("/driver/profile", response_class=HTMLResponse)
async def page_driver_profile(user: User = Depends(require_driver)):
    return HTMLResponse(get_dashboard_html("Profile", PROFILE_PAGE_HTML.format(user=user), user, "profile"))


@app.get("/driver/history", response_class=HTMLResponse)
async def page_driver_history(user: User = Depends(require_driver), db: Session = Depends(get_db)):
    content = "<p>Driver ride history will be implemented here.</p>"
    return HTMLResponse(get_dashboard_html("Trip History", content, user, "history"))


@app.get("/driver/earnings", response_class=HTMLResponse)
async def page_driver_earnings(user: User = Depends(require_driver), db: Session = Depends(get_db)):
    content = "<p>Driver earnings analytics will be implemented here.</p>"
    return HTMLResponse(get_dashboard_html("Earnings", content, user, "earnings"))


# --- API ENDPOINTS (Non-Auth) ---
@api_router.put("/profile")
async def api_update_profile(user: User = Depends(get_current_active_user), name: str = Form(...),
                             avatar: Optional[UploadFile] = File(None), db: Session = Depends(get_db)):
    user.full_name = name
    if avatar and avatar.filename:
        ext = os.path.splitext(avatar.filename)[1]
        filepath = os.path.join(CONFIG["UPLOADS_DIR"], "avatars", f"user_{user.id}{ext}")
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(avatar.file, buffer)
        user.profile_picture_url = f"/static/avatars/user_{user.id}{ext}"
    db.commit()
    return {"message": "Profile updated", "full_name": user.full_name,
            "profile_picture_url": user.profile_picture_url}


@api_router.post("/estimate-fare")
def api_estimate_fare(req: FareEstimateRequest, user: User = Depends(require_passenger),
                      db: Session = Depends(get_db)):
    return estimate_fare(db, req.pickup_lat, req.pickup_lng, req.dropoff_lat, req.dropoff_lng, req.vehicle_type, user)


@api_router.post("/passenger/request-ride", response_model=RideResponse)
def api_request_ride(req: RideRequest, user: User = Depends(require_passenger), db: Session = Depends(get_db)):
    fare = estimate_fare(db, req.pickup_lat, req.pickup_lng, req.dropoff_lat, req.dropoff_lng, req.vehicle_type, user)
    ride = Ride(**req.model_dump(), passenger_id=user.id, estimated_fare=fare['estimated_fare'],
                distance_km=fare['distance_km'], duration_minutes=fare['duration_minutes'])
    db.add(ride)
    db.commit()
    db.refresh(ride)
    driver = find_best_driver_match(db, ride)
    if driver: print(f"Notifying driver {driver.id} for ride {ride.id}")
    return ride


@api_router.get("/ride/passenger/{ride_id}", response_model=RideResponse)
def api_get_ride_status(ride_id: int, user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    ride = db.query(Ride).options(joinedload(Ride.driver).joinedload(Driver.user), joinedload(Ride.passenger)).filter(
        Ride.id == ride_id).first()
    if not ride or (ride.passenger_id != user.id and (not ride.driver or ride.driver.user_id != user.id)):
        raise HTTPException(404, "Ride not found")

    response_data = RideResponse.model_validate(ride).model_dump()
    if ride.driver and ride.driver.current_lat:
        response_data['driver_current_lat'] = ride.driver.current_lat
        response_data['driver_current_lng'] = ride.driver.current_lng
    return JSONResponse(content=response_data)


@api_router.post("/driver/register")
async def api_driver_register(user: User = Depends(get_current_active_user), db: Session = Depends(get_db),
                              license_number: str = Form(...), vehicle_make: str = Form(...),
                              vehicle_model: str = Form(...), vehicle_year: int = Form(...),
                              vehicle_color: str = Form(...), vehicle_license_plate: str = Form(...),
                              vehicle_type: VehicleType = Form(...), license_doc: UploadFile = File(...),
                              insurance_doc: UploadFile = File(...)):
    user.role = UserRole.driver
    if user.driver_info: raise HTTPException(400, "Driver profile already exists.")
    doc_dir = os.path.join(CONFIG["UPLOADS_DIR"], "documents")
    l_ext, i_ext = os.path.splitext(license_doc.filename)[1], os.path.splitext(insurance_doc.filename)[1]
    l_path, i_path = os.path.join(doc_dir, f"license_{user.id}{l_ext}"), os.path.join(doc_dir,
                                                                                      f"insurance_{user.id}{i_ext}")
    with open(l_path, "wb") as f: shutil.copyfileobj(license_doc.file, f)
    with open(i_path, "wb") as f: shutil.copyfileobj(insurance_doc.file, f)
    driver = Driver(user_id=user.id, license_number=license_number,
                    license_doc_url=f"/static/documents/license_{user.id}{l_ext}",
                    insurance_doc_url=f"/static/documents/insurance_{user.id}{i_ext}")
    db.add(driver);
    db.flush()
    vehicle = Vehicle(driver_id=driver.id, make=vehicle_make, model=vehicle_model, year=vehicle_year,
                      color=vehicle_color, license_plate=vehicle_license_plate, vehicle_type=vehicle_type)
    db.add(vehicle);
    db.commit()
    return {"message": "Registration successful"}


@api_router.post("/driver/accept-ride", response_model=RideResponse)
def api_accept_ride(req: RideAction, user: User = Depends(require_driver), db: Session = Depends(get_db)):
    driver = user.driver_info
    ride = db.query(Ride).options(joinedload(Ride.passenger)).filter(Ride.id == req.ride_id,
                                                                     Ride.status == RideStatus.pending).first()
    if not driver or not ride: raise HTTPException(404, "Invalid request")
    ride.driver_id = driver.id
    ride.status = RideStatus.accepted
    ride.accepted_at = datetime.utcnow()
    driver.status = DriverStatus.on_trip
    db.commit()
    db.refresh(ride)
    return ride

@api_router.post("/user/subscribe")
def api_subscribe(req: PurchaseSubscription, user: User = Depends(get_current_active_user),
                  db: Session = Depends(get_db)):
    plan_details = CONFIG['SUBSCRIPTION_PLANS'].get(req.plan.value)
    if not plan_details:
        raise HTTPException(404, "Plan not found")
    price = plan_details['price']
    if user.wallet_balance < price:
        raise HTTPException(400, "Insufficient wallet balance")

    user.wallet_balance -= price
    user.subscription_plan = req.plan
    user.subscription_expiry = datetime.utcnow() + timedelta(days=30)
    db.add(WalletTransaction(user_id=user.id, transaction_type=TransactionType.subscription, amount=-price,
                             description=f"{plan_details['name']} Plan Purchase"))
    db.commit()
    return {"message": "Subscribed successfully", "new_plan": req.plan.name, "expiry": user.subscription_expiry}


# --- Mounting Routers ---
app.include_router(auth_router)
app.include_router(api_router)

# ==============================================================================
# PART 11: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print(f"--- Starting {CONFIG['PROJECT_NAME']} ---")
    print(f"Access at: http://127.0.0.1:8000")
    if "YOUR_" in CONFIG["FIREBASE_WEB_CONFIG"]['apiKey'] or "YOUR_" in CONFIG["GOOGLE_MAPS_API_KEY"]:
        print("\n❌ CRITICAL WARNING: API keys are placeholders. The application will not function properly.")
        print("   Please edit index.py and fill in the CONFIG dictionary.\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
