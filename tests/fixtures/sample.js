// Test fixture: JavaScript sample code for scanner/parser/search tests.

/**
 * Fetch user data from the API.
 * @param {number} userId - The user's ID.
 * @returns {Promise<Object>} The user data.
 */
async function fetchUserData(userId) {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch user: ${response.status}`);
  }
  return response.json();
}

/**
 * Sort an array of items by a given key.
 * @param {Array} items - The items to sort.
 * @param {string} key - The key to sort by.
 * @param {string} direction - Sort direction: 'asc' or 'desc'.
 * @returns {Array} The sorted array.
 */
function sortByKey(items, key, direction = "asc") {
  return [...items].sort((a, b) => {
    if (direction === "asc") return a[key] > b[key] ? 1 : -1;
    return a[key] < b[key] ? 1 : -1;
  });
}

class EventEmitter {
  constructor() {
    this.listeners = new Map();
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
    return this;
  }

  emit(event, ...args) {
    const callbacks = this.listeners.get(event) || [];
    callbacks.forEach((cb) => cb(...args));
  }
}

module.exports = { fetchUserData, sortByKey, EventEmitter };
