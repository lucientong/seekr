"""Test fixture: Python sample code for scanner/parser/search tests."""


class UserService:
    """Service for managing user operations."""

    def __init__(self, db_connection):
        """Initialize the user service with a database connection."""
        self.db = db_connection

    def authenticate(self, username: str, password: str) -> dict:
        """Authenticate a user and return user info.

        Args:
            username: The user's login name.
            password: The user's password.

        Returns:
            A dictionary containing user information.

        Raises:
            AuthenticationError: If credentials are invalid.
        """
        user = self.db.find_user(username)
        if user and user.verify_password(password):
            return {"id": user.id, "name": user.name, "token": user.generate_token()}
        raise AuthenticationError("Invalid credentials")

    def get_user_profile(self, user_id: int) -> dict:
        """Retrieve a user's profile by their ID."""
        user = self.db.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        return user.to_dict()


def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate the discounted price."""
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class UserNotFoundError(Exception):
    """Raised when a user is not found."""
    pass
