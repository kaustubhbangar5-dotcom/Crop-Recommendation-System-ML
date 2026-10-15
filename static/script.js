/**
 * Crop Recommendation System - Client-Side Interactive Engine (9 Features, 24 Crops)
 * Handles 9-Parameter Dual Slider-Input Sync, Presets, Real-time AJAX previews, and Modal Explorer
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Parameter Sync (Range Slider <-> Number Input for all 9 features)
    const paramPairs = [
        { num: 'n_input', slider: 'n_slider' },
        { num: 'p_input', slider: 'p_slider' },
        { num: 'k_input', slider: 'k_slider' },
        { num: 'temperature_input', slider: 'temp_slider' },
        { num: 'humidity_input', slider: 'humidity_slider' },
        { num: 'ph_input', slider: 'ph_slider' },
        { num: 'rainfall_input', slider: 'rainfall_slider' },
        { num: 'soil_moisture_input', slider: 'soil_moisture_slider' },
        { num: 'sunlight_hours_input', slider: 'sunlight_hours_slider' }
    ];

    paramPairs.forEach(pair => {
        const numElem = document.getElementById(pair.num);
        const sliderElem = document.getElementById(pair.slider);

        if (numElem && sliderElem) {
            sliderElem.addEventListener('input', () => {
                numElem.value = sliderElem.value;
                triggerLivePreviewDebounced();
            });

            numElem.addEventListener('input', () => {
                sliderElem.value = numElem.value;
                triggerLivePreviewDebounced();
            });
        }
    });

    // 2. Preset Selection Handlers
    const presetButtons = document.querySelectorAll('.preset-btn');
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const n = btn.getAttribute('data-n');
            const p = btn.getAttribute('data-p');
            const k = btn.getAttribute('data-k');
            const temp = btn.getAttribute('data-temp');
            const humidity = btn.getAttribute('data-humidity');
            const ph = btn.getAttribute('data-ph');
            const rainfall = btn.getAttribute('data-rainfall');
            const moisture = btn.getAttribute('data-moisture') || '50.0';
            const sunlight = btn.getAttribute('data-sunlight') || '8.0';

            setFormValues({ n, p, k, temp, humidity, ph, rainfall, moisture, sunlight });

            btn.classList.add('pulse-active');
            setTimeout(() => btn.classList.remove('pulse-active'), 400);

            const formCard = document.getElementById('recommendation-form');
            if (formCard) {
                formCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // 3. Randomize Button
    const btnRandom = document.getElementById('btn-randomize');
    if (btnRandom) {
        btnRandom.addEventListener('click', () => {
            const randomVals = {
                n: Math.floor(Math.random() * 120) + 10,
                p: Math.floor(Math.random() * 120) + 10,
                k: Math.floor(Math.random() * 180) + 10,
                temp: (Math.random() * 25 + 15).toFixed(1),
                humidity: (Math.random() * 70 + 20).toFixed(1),
                ph: (Math.random() * 3.5 + 5.0).toFixed(1),
                rainfall: (Math.random() * 220 + 30).toFixed(1),
                moisture: (Math.random() * 60 + 25).toFixed(1),
                sunlight: (Math.random() * 5 + 5.5).toFixed(1)
            };
            setFormValues(randomVals);
        });
    }

    // 4. Reset Button
    const btnReset = document.getElementById('btn-reset');
    if (btnReset) {
        btnReset.addEventListener('click', () => {
            const defaults = {
                n: 90,
                p: 42,
                k: 43,
                temp: 20.9,
                humidity: 82.0,
                ph: 6.5,
                rainfall: 202.9,
                moisture: 85.0,
                sunlight: 6.5
            };
            setFormValues(defaults);
        });
    }

    function setFormValues(vals) {
        const mapping = [
            { id: 'n', val: vals.n },
            { id: 'p', val: vals.p },
            { id: 'k', val: vals.k },
            { id: 'temperature', val: vals.temp },
            { id: 'humidity', val: vals.humidity },
            { id: 'ph', val: vals.ph },
            { id: 'rainfall', val: vals.rainfall },
            { id: 'soil_moisture', val: vals.moisture },
            { id: 'sunlight_hours', val: vals.sunlight }
        ];

        mapping.forEach(item => {
            const num = document.getElementById(`${item.id}_input`);
            let slider = document.getElementById(`${item.id}_slider`);
            if (!slider && item.id === 'temperature') slider = document.getElementById('temp_slider');
            
            if (num) num.value = item.val;
            if (slider) slider.value = item.val;
        });

        triggerLivePreview();
    }

    // 5. Live Prediction Preview (AJAX Debounced for 9 Features)
    let debounceTimer = null;
    function triggerLivePreviewDebounced() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(triggerLivePreview, 350);
    }

    async function triggerLivePreview() {
        const previewCrop = document.getElementById('live-crop-name');
        const previewConf = document.getElementById('live-crop-conf');
        const previewIcon = document.getElementById('live-crop-icon');

        if (!previewCrop) return;

        const formData = {
            N: parseFloat(document.getElementById('n_input')?.value || 0),
            P: parseFloat(document.getElementById('p_input')?.value || 0),
            K: parseFloat(document.getElementById('k_input')?.value || 0),
            temperature: parseFloat(document.getElementById('temperature_input')?.value || 0),
            humidity: parseFloat(document.getElementById('humidity_input')?.value || 0),
            ph: parseFloat(document.getElementById('ph_input')?.value || 0),
            rainfall: parseFloat(document.getElementById('rainfall_input')?.value || 0),
            soil_moisture: parseFloat(document.getElementById('soil_moisture_input')?.value || 50.0),
            sunlight_hours: parseFloat(document.getElementById('sunlight_hours_input')?.value || 8.0)
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success && data.crop_info) {
                    previewCrop.textContent = data.crop_info.display_name || data.prediction;
                    previewConf.textContent = `Confidence: ${data.confidence}%`;
                    if (previewIcon) previewIcon.textContent = data.crop_info.icon || '🌱';
                }
            }
        } catch (err) {
            console.log('Live preview error:', err);
        }
    }

    // 6. Crop Explorer Modal
    const modal = document.getElementById('crop-modal');
    const modalClose = document.getElementById('modal-close');
    const cropCards = document.querySelectorAll('.crop-card-mini');

    cropCards.forEach(card => {
        card.addEventListener('click', () => {
            const cropName = card.getAttribute('data-crop');
            openCropModal(cropName);
        });
    });

    if (modalClose && modal) {
        modalClose.addEventListener('click', () => {
            modal.style.display = 'none';
        });

        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }

    async function openCropModal(cropName) {
        if (!modal) return;
        try {
            const res = await fetch(`/api/crop/${cropName}`);
            const data = await res.json();
            if (data.success) {
                const c = data.crop;
                document.getElementById('modal-icon').textContent = c.icon;
                document.getElementById('modal-name').textContent = c.display_name;
                document.getElementById('modal-sci').textContent = c.scientific_name;
                document.getElementById('modal-cat').textContent = c.category;
                document.getElementById('modal-desc').textContent = c.description;
                document.getElementById('modal-soil').textContent = c.soil_type;
                document.getElementById('modal-season').textContent = c.season;
                document.getElementById('modal-duration').textContent = c.growth_duration;
                document.getElementById('modal-irrigation').textContent = c.irrigation_tips;
                document.getElementById('modal-fertilizer').textContent = c.fertilizer_tips;

                modal.style.display = 'flex';
            }
        } catch (e) {
            console.error(e);
        }
    }

    // Initial Live Preview on Load
    triggerLivePreview();
});
