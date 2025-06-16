# RydePro - Advanced Ride-Hailing System

RydePro is a complete, full-stack car ordering system designed to simulate the core functionalities of services like Uber and Bolt. It is built as a single-file Python application using the FastAPI framework, providing a robust backend, an interactive frontend, and a foundation for a commercial-grade service.

This project demonstrates a wide range of modern web development practices, including role-based authentication, real-time data handling (simulated), payment gateway integration, and a scalable application architecture optimized for deployment.

## ✨ Core Features

### 🚘 Passenger Features
- **Multi-Method Authentication:** Sign up & Login via Email/Password or Google.
- **Live GPS Map:** Interactive map for setting pickup and drop-off locations.
- **Real-time Ride Booking:** Instantly request rides and receive fare estimates.
- **Dynamic Pricing:** Fare calculation includes a surge pricing algorithm based on demand.
- **Live Ride Tracking:** View your assigned driver's location in real-time on the map.
- **Ride History & Ratings:** View past trips and rate your drivers.
- **Wallet System:** Top up your in-app wallet using PayStack.
- **Subscription Plans:** Purchase monthly subscriptions for ride discounts and priority service.
- **Trip Sharing:** Generate a unique link to share your live trip location with others.
- **SOS Panic Button:** A functional SOS button to log emergencies during a ride.

### 🚖 Driver Features
- **Dedicated Driver Portal:** Separate login and registration flow for drivers.
- **Document Verification:** Upload license and insurance documents for admin approval.
- **Availability Toggle:** Go online to receive ride requests or offline when unavailable.
- **Ride Request Management:** Receive new trip requests with details and a countdown timer to accept or reject.
- **Full Ride Lifecycle Control:** Update ride status from "Arrived" to "In Progress" to "Completed".
- **Earnings Dashboard (Conceptual):** A dedicated page to view earnings.

### 🛠️ Admin Panel Features
- **Secure Admin Portal:** A separate, secure login for administrators.
- **Driver Verification:** View submitted driver applications and documents, with one-click approval or rejection.
- **System Analytics (Conceptual):** Dashboard to view key metrics like total users, drivers, and rides.
- **User & Ride Management (Conceptual):** Foundation for managing all users and trips in the system.

## 🚀 Tech Stack

- **Backend:** Python 3.10+, FastAPI
- **Database:** SQLAlchemy 2.0 with SQLite
- **Authentication:** Native Password Hashing (Passlib) & JWT, Google Social Login (Firebase)
- **Payments:** PayStack API Integration
- **Frontend:** HTML5, CSS3, JavaScript (no frameworks)
- **Deployment:** Vercel (Serverless)

## 📦 Setup and Deployment

### Prerequisites
- Python 3.10+
- A Vercel Account
- A Google Cloud/Firebase Account (for Auth and Maps)
- A PayStack Account (for Payments)
- `ngrok` (for testing webhooks locally)

### Local Development

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd rydepro_project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    - Rename `firebase-service-account.example.json` to `firebase-service-account.json` and add your Firebase service account key.
    - Open `api/main.py` and fill in all placeholder values in the `CONFIG` dictionary (Firebase Web Config, Google Maps API Key, PayStack Keys).

4.  **Run the application:**
    ```bash
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The application will be available at `http://127.0.0.1:8000`.

### 🚀 Vercel Deployment

This project is structured for easy deployment on Vercel.

1.  **Push to GitHub/GitLab/Bitbucket:** Ensure your project, with the correct folder structure (`api/main.py`, `vercel.json`, etc.), is pushed to a Git repository.

2.  **Import Project in Vercel:**
    - Log in to your Vercel dashboard.
    - Click "Add New... -> Project".
    - Import the Git repository containing your project.

3.  **Configure Environment Variables:**
    - In the Vercel project settings, go to the "Environment Variables" section.
    - Add all the secret keys from your `CONFIG` dictionary as environment variables. This is crucial for security.
        - `SECRET_KEY`
        - `FIREBASE_SERVICE_ACCOUNT_KEY_JSON` (Paste the entire content of your `.json` file here)
        - `GOOGLE_MAPS_API_KEY`
        - `PAYSTACK_SECRET_KEY`
        - `PAYSTACK_PUBLIC_KEY`
    - You will need to modify `main.py` slightly to read these from `os.environ`.

4.  **Deploy:**
    - Click the "Deploy" button. Vercel will automatically detect `vercel.json` and `requirements.txt`, build the project, and deploy it.

Of course. Here is a clear breakdown of all the essential URLs for your RydePro application, detailing where each user type should go to log in and register. This structure provides a professional and distinct user journey for each role.

You can include this information in your `README.md` or keep it as a developer reference.

---

## **RydePro Application URL Guide**

### **1. Passenger Portal**

This is the main entry point for your customers (riders).

*   **Homepage / Main Login:**
    *   **URL:** `http://127.0.0.1:8000/`
    *   **Description:** This is the primary landing page. Unauthenticated users will see the main marketing page. If they click the "Login / Sign Up" button, they are taken to the passenger login/registration form.

*   **Passenger Login Page:**
    *   **URL:** `http://127.0.0.1:8000/login`
    *   **Description:** This page features a tabbed interface for passengers to log in either via **Email/Password** or **Google (Social Login)**.

*   **Passenger Registration Page:**
    *   **URL:** `http://127.0.0.1:8000/register`
    *   **Description:** A direct link to the sign-up form for new passengers. Users are redirected here from the login page if they don't have an account.

### **2. Driver Portal**

This is the dedicated portal for your drivers. It has a separate entry point to distinguish it from the passenger flow.

*   **Main Driver Portal Page:**
    *   **URL:** `http://127.0.0.1:8000/driver/portal`
    *   **Description:** This is the landing page specifically for drivers. It should contain information relevant to drivers and provides clear links for them to log in or start the application process.

*   **Driver Login Page:**
    *   **URL:** `http://127.0.0.1:8000/driver/login`
    *   **Description:** The login form exclusively for existing, approved drivers. The backend will verify that the user logging in has the `driver` role.

*   **Driver Registration / Application Page:**
    *   **URL:** `http://127.0.0.1:8000/driver/register`
    *   **Description:** This is the start of the multi-step driver onboarding process. New driver candidates land here to fill out their vehicle and document information. It's linked from the main driver portal.

### **3. Admin Portal**

This portal is for system administrators and should not be publicly linked on the main site for security reasons (security through obscurity).

*   **Admin Login Page:**
    *   **URL:** `http://127.0.0.1:8000/site/admin/login`
    *   **Description:** A dedicated, non-discoverable login page for administrators. The backend will verify that the user logging in has the `admin` role.

*   **Admin Registration (Implicit):**
    *   **URL:** `http://127.0.0.1:8000/register`
    *   **Description:** There is no separate registration form for admins. The first admin account is created by registering through the **main passenger registration page** using the specific email address `admin@rydepro.com`. The backend logic automatically assigns the `admin` role to this email.

---

### **Summary Table**

| User Role | Task | URL |
| :--- | :--- | :--- |
| **Passenger** | Login | `http://127.0.0.1:8000/login` |
| **Passenger** | Register | `http://127.0.0.1:8000/register` |
| **Driver** | Login | `http://127.0.0.1:8000/driver/login` |
| **Driver** | Register (Apply) | `http://127.0.0.1:8000/driver/register` |
| **Admin** | Login | `http://127.0.0.1:8000/site/admin/login` |
| **Admin** | Register (First Time) | `http://127.0.0.1:8000/register` (using `admin@rydepro.com`) |
---
