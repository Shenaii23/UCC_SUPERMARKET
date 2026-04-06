// Global state
let cart = JSON.parse(localStorage.getItem('cart')) || [];
let products = [];
let currentPage = 1;
let itemsPerPage = 8;
let currentCategory = 'all';
let currentSort = 'name-asc';

// Sample product data
const sampleProducts = [
    { id: 1, name: 'Fresh Apples', category: 'fruits', price: 2.99, image: '🍎', description: 'Fresh red apples' },
    { id: 2, name: 'Organic Bananas', category: 'fruits', price: 1.99, image: '🍌', description: 'Organic bananas' },
    { id: 3, name: 'Baby Spinach', category: 'vegetables', price: 3.49, image: '🥬', description: 'Fresh baby spinach' },
    { id: 4, name: 'Whole Milk', category: 'dairy', price: 3.99, image: '🥛', description: 'Fresh whole milk' },
    { id: 5, name: 'Sourdough Bread', category: 'bakery', price: 4.99, image: '🍞', description: 'Fresh baked sourdough' },
    { id: 6, name: 'Free Range Eggs', category: 'dairy', price: 5.99, image: '🥚', description: 'Free range eggs' },
    { id: 7, name: 'Chicken Breast', category: 'meat', price: 8.99, image: '🍗', description: 'Boneless chicken breast' },
    { id: 8, name: 'Atlantic Salmon', category: 'seafood', price: 12.99, image: '🐟', description: 'Fresh Atlantic salmon' },
    { id: 9, name: 'Greek Yogurt', category: 'dairy', price: 4.49, image: '🥄', description: 'Plain Greek yogurt' },
    { id: 10, name: 'Avocados', category: 'fruits', price: 2.49, image: '🥑', description: 'Ripe avocados' },
    { id: 11, name: 'Organic Carrots', category: 'vegetables', price: 2.29, image: '🥕', description: 'Organic carrots' },
    { id: 12, name: 'Ground Beef', category: 'meat', price: 7.99, image: '🥩', description: 'Lean ground beef' },
    { id: 13, name: 'Orange Juice', category: 'beverages', price: 4.99, image: '🧃', description: 'Fresh squeezed orange juice' },
    { id: 14, name: 'Cheddar Cheese', category: 'dairy', price: 6.49, image: '🧀', description: 'Aged cheddar cheese' },
    { id: 15, name: 'Pasta', category: 'pantry', price: 1.99, image: '🍝', description: 'Italian pasta' },
    { id: 16, name: 'Tomatoes', category: 'vegetables', price: 3.29, image: '🍅', description: 'Roma tomatoes' }
];

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    initializeChatbot();
    updateCartCount();
    
    // Load products if on products page
    if (document.querySelector('.products-grid')) {
        loadProducts();
        initializeFilters();
    }
    
    // Load cart if on cart page
    if (document.querySelector('.cart-items')) {
        loadCart();
    }
    
    // Add event listeners for cart page
    if (document.querySelector('.checkout-btn')) {
        initializeCheckout();
    }
});

// Chatbot functionality
function initializeChatbot() {
    const chatbotButton = document.querySelector('.chatbot-button');
    const chatbotContainer = document.querySelector('.chatbot-container');
    const closeChat = document.querySelector('.close-chat');
    const sendButton = document.querySelector('.send-message');
    const chatInput = document.querySelector('.chat-input');
    const chatMessages = document.querySelector('.chat-messages');
    const suggestedQuestions = document.querySelectorAll('.suggested-question');
    
    if (!chatbotButton || !chatbotContainer) return;
    
    // Toggle chat
    chatbotButton.addEventListener('click', () => {
        chatbotContainer.classList.toggle('hidden');
        if (!chatbotContainer.classList.contains('hidden')) {
            addBotMessage("Hi! I'm your UCC Supermarket assistant. How can I help you today?");
        }
    });
    
    if (closeChat) {
        closeChat.addEventListener('click', () => {
            chatbotContainer.classList.add('hidden');
        });
    }
    
    // Send message
    if (sendButton && chatInput) {
        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }
    
    // Suggested questions
    if (suggestedQuestions) {
        suggestedQuestions.forEach(question => {
            question.addEventListener('click', () => {
                const text = question.textContent;
                if (chatInput) chatInput.value = text;
                sendMessage();
            });
        });
    }
    
    function sendMessage(text = null) {
        if (!chatInput || !chatMessages) return;
        const message = text !== null ? text : chatInput.value.trim();
        if (!message) return;
        
        addUserMessage(message);
        if (text === null) {
            chatInput.value = '';
        }
        
        // Show thinking indicator
        showThinking();
        
        // Simulate bot response (replace with actual API call)
        setTimeout(() => {
            removeThinking();
            processBotResponse(message);
        }, 1000);
    }
    
    function addUserMessage(text) {
        if (!chatMessages) return;
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function escapeHTML(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function parseSelectionItems(text) {
        const normalized = text
            .replace(/\r\n?/g, '\n')
            .replace(/•\s*/g, '\n- ')
            .replace(/(^|\n)-\s*/g, '\n- ');
        const lines = normalized.split(/\n+/);
        const items = lines
            .map(line => line.trim())
            .filter(line => line.startsWith('- '))
            .map(line => line.slice(2).trim())
            .filter(line => line.length > 0);
        return items.length ? items : null;
    }

    function createSelectionPanel(items) {
        const panel = document.createElement('div');
        panel.className = 'product-selection';

        const instructions = document.createElement('div');
        instructions.className = 'selection-instructions';
        instructions.textContent = 'Select one or more items and send them to the assistant.';
        panel.appendChild(instructions);

        const list = document.createElement('div');
        list.className = 'product-selection-list';

        items.forEach((item, idx) => {
            const label = document.createElement('label');
            label.className = 'product-option';
            label.htmlFor = `product-option-${idx}`;
            label.innerHTML = `
                <input type="checkbox" id="product-option-${idx}" data-item="${escapeHTML(item)}">
                <span>${escapeHTML(item)}</span>
            `;
            list.appendChild(label);
        });
        panel.appendChild(list);

        const actions = document.createElement('div');
        actions.className = 'product-selection-actions';

        const submitButton = document.createElement('button');
        submitButton.type = 'button';
        submitButton.textContent = 'Send selection';
        submitButton.className = 'selection-send-btn';
        submitButton.addEventListener('click', () => {
            const checked = Array.from(panel.querySelectorAll('input[type="checkbox"]:checked'));
            if (!checked.length) return;
            const selection = checked.map(input => input.dataset.item).join(', ');
            sendMessage(selection);
        });

        const clearButton = document.createElement('button');
        clearButton.type = 'button';
        clearButton.textContent = 'Clear';
        clearButton.className = 'selection-clear-btn';
        clearButton.addEventListener('click', () => {
            panel.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
        });

        actions.appendChild(clearButton);
        actions.appendChild(submitButton);
        panel.appendChild(actions);

        return panel;
    }

    function formatBotMessage(text) {
        const safeText = escapeHTML(text)
            .replace(/\r\n?/g, '\n')
            .replace(/(^|\n)•\s*/g, '$1- ')
            .replace(/(^|\n)-\s*/g, '$1- ');
        if (typeof marked !== 'undefined') {
            return marked.parse(safeText);
        }
        return safeText.replace(/\n/g, '<br>');
    }

    function addBotMessage(text) {
        if (!chatMessages) return;
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.innerHTML = formatBotMessage(text);

        const selectionItems = parseSelectionItems(text);
        if (selectionItems) {
            const panel = createSelectionPanel(selectionItems);
            messageDiv.appendChild(panel);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function showThinking() {
        if (!chatMessages) return;
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'message bot thinking';
        thinkingDiv.textContent = 'Thinking';
        thinkingDiv.id = 'thinking-message';
        chatMessages.appendChild(thinkingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function removeThinking() {
        const thinking = document.getElementById('thinking-message');
        if (thinking) thinking.remove();
    }
    
    function processBotResponse(message) {
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
            addBotMessage("Hello! How can I assist you with your shopping today?");
        } 
        else if (lowerMessage.includes('cart') || lowerMessage.includes('basket')) {
            addBotMessage("I can help you add items to your cart! Just tell me what you'd like to buy.");
        }
        else if (lowerMessage.includes('price') || lowerMessage.includes('cost')) {
            addBotMessage("You can check prices on our products page. What specific item are you interested in?");
        }
        else if (lowerMessage.includes('help')) {
            addBotMessage("I can help you with:\n- Finding products\n- Adding items to cart\n- Checking prices\n- Placing orders");
        }
        else if (lowerMessage.includes('add')) {
            // Extract product name and add to cart
            const productName = message.replace(/add|to|cart|please/gi, '').trim();
            if (productName) {
                const product = sampleProducts.find(p => 
                    p.name.toLowerCase().includes(productName.toLowerCase())
                );
                
                if (product) {
                    addToCart(product.id);
                    addBotMessage(`✅ I've added ${product.name} to your cart! You can view your cart by clicking the cart icon.`);
                    updateCartCount();
                } else {
                    addBotMessage(`I couldn't find "${productName}". Could you be more specific?`);
                }
            }
        }
        else if (lowerMessage.includes('show') || lowerMessage.includes('my cart')) {
            addBotMessage(`You have ${cart.length} items in your cart. Total: $${calculateCartTotal().toFixed(2)}`);
        }
        else {
            addBotMessage("I understand you need help. Could you be more specific? Try asking about products, prices, or adding items to cart.");
        }
    }
}

// Cart functions
function addToCart(productId, quantity = 1) {
    const product = sampleProducts.find(p => p.id === productId);
    if (!product) return;
    
    const existingItem = cart.find(item => item.id === productId);
    
    if (existingItem) {
        existingItem.quantity += quantity;
    } else {
        cart.push({
            ...product,
            quantity: quantity
        });
    }
    
    saveCart();
    updateCartCount();
    
    // Show added animation
    showAddedToCart(product.name);
}

function removeFromCart(productId) {
    cart = cart.filter(item => item.id !== productId);
    saveCart();
    updateCartCount();
    loadCart(); // Reload cart display
}

function updateQuantity(productId, change) {
    const item = cart.find(item => item.id === productId);
    if (!item) return;
    
    item.quantity += change;
    
    if (item.quantity <= 0) {
        removeFromCart(productId);
    } else {
        saveCart();
        loadCart(); // Reload cart display
    }
}

function saveCart() {
    localStorage.setItem('cart', JSON.stringify(cart));
}

function updateCartCount() {
    const cartCounts = document.querySelectorAll('.cart-count');
    const totalItems = cart.reduce((sum, item) => sum + item.quantity, 0);
    
    cartCounts.forEach(count => {
        count.textContent = totalItems;
    });
}

function calculateCartTotal() {
    return cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
}

function showAddedToCart(productName) {
    // You could implement a toast notification here
    console.log(`Added ${productName} to cart`);
}

// Load cart on cart page
function loadCart() {
    const cartContainer = document.querySelector('.cart-items');
    const summaryContainer = document.querySelector('.cart-summary');
    
    if (!cartContainer || !summaryContainer) return;
    
    if (cart.length === 0) {
        cartContainer.innerHTML = `
            <div class="empty-cart">
                <i class="fas fa-shopping-cart"></i>
                <p>Your cart is empty</p>
                <a href="/products.html" class="btn">Start Shopping</a>
            </div>
        `;
        updateCartSummary(0);
        return;
    }
    
    let cartHTML = '';
    
    cart.forEach(item => {
        cartHTML += `
            <div class="cart-item" data-id="${item.id}">
                <div class="cart-item-image">${item.image}</div>
                <div class="cart-item-details">
                    <h3>${item.name}</h3>
                    <p class="cart-item-price">$${item.price.toFixed(2)} each</p>
                </div>
                <div class="cart-item-quantity">
                    <button class="quantity-btn decrease" onclick="updateQuantity(${item.id}, -1)" ${item.quantity <= 1 ? 'disabled' : ''}>-</button>
                    <span class="quantity-value">${item.quantity}</span>
                    <button class="quantity-btn increase" onclick="updateQuantity(${item.id}, 1)">+</button>
                </div>
                <button class="remove-item" onclick="removeFromCart(${item.id})">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
    });
    
    cartContainer.innerHTML = cartHTML;
    updateCartSummary(calculateCartTotal());
}

function updateCartSummary(total) {
    const subtotalEl = document.querySelector('.subtotal');
    const totalEl = document.querySelector('.total span:last-child');
    
    if (subtotalEl) {
        subtotalEl.textContent = `$${total.toFixed(2)}`;
    }
    
    if (totalEl) {
        totalEl.textContent = `$${total.toFixed(2)}`;
    }
}

// Checkout modal
function initializeCheckout() {
    const checkoutBtn = document.querySelector('.checkout-btn');
    const modal = document.getElementById('checkoutModal');
    const closeModal = document.querySelector('.close-modal');
    const continueShopping = document.querySelector('.continue-shopping');
    const viewCart = document.querySelector('.view-cart');
    
    if (!checkoutBtn || !modal) return;
    
    checkoutBtn.addEventListener('click', () => {
        if (cart.length === 0) {
            alert('Your cart is empty!');
            return;
        }
        modal.classList.add('show');
    });
    
    if (closeModal) {
        closeModal.addEventListener('click', () => {
            modal.classList.remove('show');
        });
    }
    
    if (continueShopping) {
        continueShopping.addEventListener('click', () => {
            modal.classList.remove('show');
        });
    }
    
    if (viewCart) {
        viewCart.addEventListener('click', () => {
            modal.classList.remove('show');
            window.location.href = '/cart.html';
        });
    }
    
    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    });
}

// Products page functions
function loadProducts() {
    const grid = document.querySelector('.products-grid');
    if (!grid) return;
    
    products = sampleProducts;
    filterAndSortProducts();
}

function filterAndSortProducts() {
    let filteredProducts = [...products];
    
    // Filter by category
    if (currentCategory !== 'all') {
        filteredProducts = filteredProducts.filter(p => p.category === currentCategory);
    }
    
    // Sort products
    filteredProducts.sort((a, b) => {
        switch(currentSort) {
            case 'name-asc':
                return a.name.localeCompare(b.name);
            case 'name-desc':
                return b.name.localeCompare(a.name);
            case 'price-asc':
                return a.price - b.price;
            case 'price-desc':
                return b.price - a.price;
            default:
                return 0;
        }
    });
    
    // Paginate
    const start = (currentPage - 1) * itemsPerPage;
    const paginatedProducts = filteredProducts.slice(start, start + itemsPerPage);
    
    displayProducts(paginatedProducts);
    updatePagination(filteredProducts.length);
}

function displayProducts(productsToDisplay) {
    const grid = document.querySelector('.products-grid');
    if (!grid) return;
    
    if (productsToDisplay.length === 0) {
        grid.innerHTML = '<p class="text-center">No products found</p>';
        return;
    }
    
    let productsHTML = '';
    
    productsToDisplay.forEach(product => {
        productsHTML += `
            <div class="product-card">
                <div class="product-image">${product.image}</div>
                <div class="product-info">
                    <h3>${product.name}</h3>
                    <span class="product-category">${product.category}</span>
                    <p class="product-price">$${product.price.toFixed(2)}</p>
                    <button class="add-to-cart-btn" onclick="addToCart(${product.id})">
                        <i class="fas fa-cart-plus"></i>
                        Add to Cart
                    </button>
                </div>
            </div>
        `;
    });
    
    grid.innerHTML = productsHTML;
}

function initializeFilters() {
    const categoryFilter = document.getElementById('category');
    const sortFilter = document.getElementById('sort');
    
    if (categoryFilter) {
        categoryFilter.addEventListener('change', (e) => {
            currentCategory = e.target.value;
            currentPage = 1;
            filterAndSortProducts();
        });
    }
    
    if (sortFilter) {
        sortFilter.addEventListener('change', (e) => {
            currentSort = e.target.value;
            filterAndSortProducts();
        });
    }
}

function updatePagination(totalItems) {
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    const paginationContainer = document.querySelector('.pagination');
    
    if (!paginationContainer) return;
    
    if (totalPages <= 1) {
        paginationContainer.innerHTML = '';
        return;
    }
    
    let paginationHTML = '';
    
    // Previous button
    paginationHTML += `
        <button onclick="changePage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>
            <i class="fas fa-chevron-left"></i>
        </button>
    `;
    
    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || (i >= currentPage - 1 && i <= currentPage + 1)) {
            paginationHTML += `
                <button onclick="changePage(${i})" class="${i === currentPage ? 'active' : ''}">
                    ${i}
                </button>
            `;
        } else if (i === currentPage - 2 || i === currentPage + 2) {
            paginationHTML += `<button disabled>...</button>`;
        }
    }
    
    // Next button
    paginationHTML += `
        <button onclick="changePage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>
            <i class="fas fa-chevron-right"></i>
        </button>
    `;
    
    paginationContainer.innerHTML = paginationHTML;
}

function changePage(newPage) {
    currentPage = newPage;
    filterAndSortProducts();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Make functions global
window.addToCart = addToCart;
window.removeFromCart = removeFromCart;
window.updateQuantity = updateQuantity;
window.changePage = changePage;