document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const locationSelect = document.getElementById("location-select");
    const resultContainer = document.getElementById("result-container");
    const priceNumber = document.getElementById("price-number");
    const priceRupees = document.getElementById("price-rupees");
    const shapList = document.getElementById("shap-list");
    const errorContainer = document.getElementById("error-container");
    const errorMessage = document.getElementById("error-message");
    const submitBtn = document.getElementById("submit-btn");

    // Initialize locations list from API
    async function loadLocations() {
        try {
            const response = await fetch("/locations");
            if (!response.ok) {
                throw new Error("Failed to fetch locations list.");
            }
            const data = await response.json();
            
            // Clear loading placeholder
            locationSelect.innerHTML = '<option value="" disabled selected>Select a location</option>';
            
            // Populate select dropdown
            data.locations.forEach(loc => {
                const opt = document.createElement("option");
                opt.value = loc;
                opt.textContent = loc;
                locationSelect.appendChild(opt);
            });
        } catch (err) {
            console.error(err);
            showError("Could not connect to API server. Please make sure the backend is running.");
        }
    }

    // Format lakhs to formatted Indian Rupee currency representation
    function formatRupees(lakhs) {
        const rupees = Math.round(lakhs * 100000);
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            maximumFractionDigits: 0
        }).format(rupees);
    }

    // Map raw feature names to human-readable labels
    function getFeatureLabel(rawFeature, selectedLocation) {
        if (rawFeature === "total_sqft") {
            return "Property Size (Square Feet)";
        }
        if (rawFeature === "bath") {
            return "Number of Bathrooms";
        }
        if (rawFeature === "bhk") {
            return "Number of Bedrooms (BHK)";
        }
        if (rawFeature.startsWith("location_")) {
            const loc = rawFeature.replace("location_", "");
            return `Location: ${loc}`;
        }
        if (rawFeature === "location") {
            return `Location: ${selectedLocation}`;
        }
        // Fallback
        return rawFeature;
    }

    // Show error alert
    function showError(msg) {
        errorMessage.textContent = msg;
        errorContainer.classList.remove("hidden");
        resultContainer.classList.add("hidden");
    }

    // Clear alerts
    function clearAlerts() {
        errorContainer.classList.add("hidden");
    }

    // Handle form submit
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearAlerts();
        
        // Change button state
        submitBtn.disabled = true;
        submitBtn.textContent = "Calculating estimated price...";

        const payload = {
            location: locationSelect.value,
            sqft: parseFloat(document.getElementById("sqft-input").value),
            bhk: parseInt(document.getElementById("bhk-select").value),
            bath: parseInt(document.getElementById("bath-select").value)
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                const detail = errData.detail;
                const errorText = Array.isArray(detail) ? detail.map(d => d.msg).join(", ") : detail;
                throw new Error(errorText || "API prediction error.");
            }

            const data = await response.json();
            
            // Render Price
            const predicted = data.predicted_price;
            priceNumber.textContent = predicted.toFixed(2);
            priceRupees.textContent = formatRupees(predicted);
            
            // Render SHAP explainability cards
            shapList.innerHTML = "";
            
            if (data.shap_contributions && data.shap_contributions.length > 0) {
                data.shap_contributions.forEach(item => {
                    const shapItem = document.createElement("div");
                    shapItem.className = "shap-item";
                    
                    const labelSpan = document.createElement("span");
                    labelSpan.className = "shap-feature-name";
                    labelSpan.textContent = getFeatureLabel(item.feature, payload.location);
                    
                    const valBadge = document.createElement("span");
                    const isPositive = item.contribution > 0;
                    valBadge.className = `shap-badge ${isPositive ? 'positive' : 'negative'}`;
                    
                    const absVal = Math.abs(item.contribution).toFixed(2);
                    valBadge.textContent = `${isPositive ? '+' : '-'}${absVal} Lakhs`;
                    
                    shapItem.appendChild(labelSpan);
                    shapItem.appendChild(valBadge);
                    shapList.appendChild(shapItem);
                });
            } else {
                shapList.innerHTML = '<p class="explain-text">No feature influence could be calculated for this listing.</p>';
            }

            // Show result block
            resultContainer.classList.remove("hidden");
        } catch (err) {
            console.error(err);
            showError(`Error: ${err.message}`);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Calculate Estimated Price";
        }
    });

    // Initial Load
    loadLocations();
});
