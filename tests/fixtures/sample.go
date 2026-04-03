// Test fixture: Go sample code for scanner/parser/search tests.

package main

import (
	"errors"
	"fmt"
)

// User represents a system user.
type User struct {
	ID       int
	Name     string
	Email    string
	IsActive bool
}

// UserRepository provides access to user storage.
type UserRepository interface {
	FindByID(id int) (*User, error)
	FindByEmail(email string) (*User, error)
	Save(user *User) error
	Delete(id int) error
}

// AuthService handles user authentication.
type AuthService struct {
	repo UserRepository
}

// NewAuthService creates a new authentication service.
func NewAuthService(repo UserRepository) *AuthService {
	return &AuthService{repo: repo}
}

// Authenticate validates user credentials and returns a token.
func (s *AuthService) Authenticate(email, password string) (string, error) {
	user, err := s.repo.FindByEmail(email)
	if err != nil {
		return "", fmt.Errorf("authentication failed: %w", err)
	}
	if user == nil {
		return "", errors.New("user not found")
	}
	// TODO: verify password hash
	return fmt.Sprintf("token-%d", user.ID), nil
}

// CalculateAverage computes the average of a slice of numbers.
func CalculateAverage(numbers []float64) (float64, error) {
	if len(numbers) == 0 {
		return 0, errors.New("cannot calculate average of empty slice")
	}
	sum := 0.0
	for _, n := range numbers {
		sum += n
	}
	return sum / float64(len(numbers)), nil
}
