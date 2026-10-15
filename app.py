"""
Crop Recommendation System - Flask Web Application (9 Features, 24 Crop Classes)
Provides real-time machine learning crop recommendations based on soil nutrients and climate parameters.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'crop_recommendation_ml_secret_key'

# -------------------------------------------------------------
# 1. LOAD TRAINED MACHINE LEARNING MODEL
# -------------------------------------------------------------
MODEL_FILE = 'crop_model.pkl'
model_bundle = None

def load_ml_model():
    global model_bundle
    if os.path.exists(MODEL_FILE):
        try:
            model_bundle = joblib.load(MODEL_FILE)
            print(f"[*] ML model loaded successfully from '{MODEL_FILE}'.")
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            model_bundle = None
    else:
        print(f"[!] Warning: '{MODEL_FILE}' not found. Please run train_model.py first.")

load_ml_model()

# -------------------------------------------------------------
# 2. CROP KNOWLEDGE BASE & AGRONOMIC METADATA (24 CROPS)
# -------------------------------------------------------------
CROP_DETAILS = {
    'rice': {
        'display_name': 'Rice',
        'scientific_name': 'Oryza sativa',
        'category': 'Cereal Grain',
        'icon': '🌾',
        'tagline': 'High water, warm climate staple cereal.',
        'description': 'Rice is the most widely consumed staple food for a large part of the world\'s human population. It requires substantial water availability, warm temperatures, and nutrient-rich soil with high water retention.',
        'soil_type': 'Heavy clay or clayey loam with excellent water holding capacity.',
        'season': 'Kharif (Monsoon) / Summer',
        'growth_duration': '120 - 150 days',
        'optimal_ranges': {'N': '60 - 100 kg/ha', 'P': '35 - 60 kg/ha', 'K': '35 - 45 kg/ha', 'temp': '20 - 27 °C', 'humidity': '80 - 85 %', 'ph': '5.0 - 7.5', 'rainfall': '180 - 300 mm', 'soil_moisture': '75 - 95 %', 'sunlight_hours': '5.5 - 7.5 hrs/day'},
        'irrigation_tips': 'Maintain standing water (2-5 cm) during tillering and flowering stages. Avoid drought stress during panicle initiation.',
        'fertilizer_tips': 'Apply Nitrogen in 3 split doses (basal, tillering, panicle initiation). Supplement Zinc if soil is alkaline.',
        'economic_value': 'High commercial demand, strong market pricing, and stable food security crop.'
    },
    'wheat': {
        'display_name': 'Wheat',
        'scientific_name': 'Triticum aestivum',
        'category': 'Rabi Cereal Grain',
        'icon': '🍞',
        'tagline': 'Premier cool-season staple cereal feeding billions globally.',
        'description': 'Wheat is the world\'s leading cereal grain crop, grown predominantly in temperate and cool subtropical winter seasons. It thrives in loamy soils with moderate moisture and bright sunny days during grain filling.',
        'soil_type': 'Well-drained fertile loam and clayey loam soils with good structure.',
        'season': 'Rabi (Winter season)',
        'growth_duration': '110 - 140 days',
        'optimal_ranges': {'N': '80 - 130 kg/ha', 'P': '40 - 70 kg/ha', 'K': '30 - 55 kg/ha', 'temp': '12 - 24 °C', 'humidity': '45 - 70 %', 'ph': '6.0 - 7.5', 'rainfall': '45 - 100 mm', 'soil_moisture': '40 - 65 %', 'sunlight_hours': '7.0 - 9.5 hrs/day'},
        'irrigation_tips': 'Critical irrigation at Crown Root Initiation (CRI at 21 days), tillering, boot stage, and grain filling stages.',
        'fertilizer_tips': 'Apply balanced NPK (120:60:40) with split nitrogen. Supplement Sulfur and Zinc for higher gluten content.',
        'economic_value': 'Global staple with guaranteed minimum support price (MSP), massive flour milling and baking industry.'
    },
    'sugarcane': {
        'display_name': 'Sugarcane',
        'scientific_name': 'Saccharum officinarum',
        'category': 'Industrial Cash Crop',
        'icon': '🎋',
        'tagline': 'High-biomass tropical giant driving sugar and biofuel energy.',
        'description': 'Sugarcane is a tall perennial grass cultivated in tropical and subtropical regions for sugar and ethanol biofuel. It is a heavy feeder requiring high sunlight, rich soil nutrients, and ample irrigation.',
        'soil_type': 'Deep, fertile, well-drained loamy or clayey loam soils with good water retention.',
        'season': 'Year-round planting (Spring / Autumn)',
        'growth_duration': '10 - 14 months',
        'optimal_ranges': {'N': '120 - 180 kg/ha', 'P': '50 - 90 kg/ha', 'K': '70 - 125 kg/ha', 'temp': '24 - 38 °C', 'humidity': '65 - 95 %', 'ph': '6.0 - 8.0', 'rainfall': '140 - 260 mm', 'soil_moisture': '65 - 92 %', 'sunlight_hours': '7.5 - 10.5 hrs/day'},
        'irrigation_tips': 'High water requirement. Furrow or drip irrigation every 8-12 days during grand growth phase. Withhold water 15 days before harvest.',
        'fertilizer_tips': 'Heavy nitrogen and potash consumer. Apply potash at 90-120 days to enhance sucrose recovery percentage.',
        'economic_value': 'Multi-revenue from refined sugar, jaggery, molasses, bagasse co-generation, and ethanol biofuel.'
    },
    'maize': {
        'display_name': 'Maize (Corn)',
        'scientific_name': 'Zea mays',
        'category': 'Cereal & Fodder',
        'icon': '🌽',
        'tagline': 'Versatile queen of cereals with high photosynthetic efficiency.',
        'description': 'Maize is a versatile cereal grown across diverse agro-climatic conditions. It is heavily utilized for food, poultry feed, and industrial starch production.',
        'soil_type': 'Well-drained deep loamy and silt-loam soils rich in organic matter.',
        'season': 'Kharif, Rabi, and Spring',
        'growth_duration': '90 - 120 days',
        'optimal_ranges': {'N': '60 - 100 kg/ha', 'P': '35 - 60 kg/ha', 'K': '15 - 25 kg/ha', 'temp': '18 - 27 °C', 'humidity': '55 - 75 %', 'ph': '5.5 - 7.5', 'rainfall': '60 - 110 mm', 'soil_moisture': '45 - 68 %', 'sunlight_hours': '7.5 - 9.5 hrs/day'},
        'irrigation_tips': 'Critical irrigation stages are tasseling and silking. Sensitive to waterlogging; ensure proper field drainage.',
        'fertilizer_tips': 'High nitrogen consumer. Apply balanced NPK with sulfur and zinc for maximum cob development.',
        'economic_value': 'Fast turnover, dual revenue from grain and cattle fodder, high industrial demand.'
    },
    'chickpea': {
        'display_name': 'Chickpea (Gram)',
        'scientific_name': 'Cicer arietinum',
        'category': 'Pulse / Legume',
        'icon': '🧆',
        'tagline': 'Protein-rich legume improving soil nitrogen balance.',
        'description': 'Chickpea is a major cool-season pulse crop that fixes atmospheric nitrogen, enriching soil fertility naturally. It thrives in mild winter conditions.',
        'soil_type': 'Deep, well-drained sandy loam or clay loam with good aeration.',
        'season': 'Rabi (Winter)',
        'growth_duration': '90 - 115 days',
        'optimal_ranges': {'N': '20 - 60 kg/ha', 'P': '55 - 80 kg/ha', 'K': '75 - 85 kg/ha', 'temp': '17 - 21 °C', 'humidity': '14 - 20 %', 'ph': '6.0 - 8.5', 'rainfall': '65 - 95 mm', 'soil_moisture': '20 - 36 %', 'sunlight_hours': '7.0 - 9.2 hrs/day'},
        'irrigation_tips': 'Requires light irrigation at pre-flowering and pod development. Excessive moisture causes wilt disease.',
        'fertilizer_tips': 'Requires lower nitrogen due to Rhizobium nodules. Emphasize phosphorus and potassium for deep root formation.',
        'economic_value': 'High export potential, strong domestic protein demand, low chemical fertilizer cost.'
    },
    'kidneybeans': {
        'display_name': 'Kidney Beans (Rajma)',
        'scientific_name': 'Phaseolus vulgaris',
        'category': 'Pulse / Legume',
        'icon': '🫘',
        'tagline': 'High-value legume requiring cool climate and rich loams.',
        'description': 'Kidney beans are popular high-protein legumes grown in temperate and subtropical hill regions. They require moderate humidity and well-aerated fertile soils.',
        'soil_type': 'Rich, well-aerated light loam with neutral to slightly acidic pH.',
        'season': 'Rabi in plains / Kharif in hills',
        'growth_duration': '100 - 120 days',
        'optimal_ranges': {'N': '15 - 40 kg/ha', 'P': '55 - 80 kg/ha', 'K': '15 - 25 kg/ha', 'temp': '15 - 24 °C', 'humidity': '18 - 25 %', 'ph': '5.5 - 6.0', 'rainfall': '60 - 150 mm', 'soil_moisture': '32 - 52 %', 'sunlight_hours': '6.2 - 8.5 hrs/day'},
        'irrigation_tips': 'Keep soil moist but never waterlogged. Critical watering needed at flowering and pod filling.',
        'fertilizer_tips': 'Unlike other legumes, kidney beans fix less nitrogen; apply moderate starter nitrogen and ample phosphorus.',
        'economic_value': 'Premium market price in urban centers, excellent nutritional value.'
    },
    'pigeonpeas': {
        'display_name': 'Pigeon Peas (Arhar / Tur)',
        'scientific_name': 'Cajanus cajan',
        'category': 'Pulse / Legume',
        'icon': '🌱',
        'tagline': 'Deep-rooted drought-resilient pulse for warm climates.',
        'description': 'Pigeon pea is an essential source of dietary protein in tropical regions. Its deep taproot system makes it exceptionally tolerant to drought and heat.',
        'soil_type': 'Deep, well-drained loam or alluvial soils. Tolerates slightly alkaline conditions.',
        'season': 'Kharif',
        'growth_duration': '150 - 180 days',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '55 - 80 kg/ha', 'K': '15 - 25 kg/ha', 'temp': '20 - 35 °C', 'humidity': '40 - 70 %', 'ph': '5.0 - 7.5', 'rainfall': '90 - 200 mm', 'soil_moisture': '28 - 48 %', 'sunlight_hours': '7.0 - 9.2 hrs/day'},
        'irrigation_tips': 'Generally rainfed. Supplementary irrigation during flowering significantly boosts pod set.',
        'fertilizer_tips': 'Inoculate seeds with Rhizobium culture. Apply phosphorus at sowing for robust root system.',
        'economic_value': 'Staple pulse with steady market demand, wood utilized as domestic bio-fuel.'
    },
    'mothbeans': {
        'display_name': 'Moth Beans',
        'scientific_name': 'Vigna aconitifolia',
        'category': 'Pulse / Arid Crop',
        'icon': '🥜',
        'tagline': 'Extremely drought-hardy arid region legume.',
        'description': 'Moth bean is one of the most drought-resistant legumes, making it ideal for arid and semi-arid agro-ecological zones with very low rainfall.',
        'soil_type': 'Sandy loam, light sandy soils, and marginal desert lands.',
        'season': 'Kharif',
        'growth_duration': '75 - 90 days',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '35 - 60 kg/ha', 'K': '15 - 25 kg/ha', 'temp': '24 - 32 °C', 'humidity': '40 - 65 %', 'ph': '4.5 - 9.0', 'rainfall': '30 - 75 mm', 'soil_moisture': '12 - 28 %', 'sunlight_hours': '8.5 - 10.5 hrs/day'},
        'irrigation_tips': 'Thrives in low rainfall; avoid over-irrigation as it easily causes root rot.',
        'fertilizer_tips': 'Requires minimal fertilization. Light phosphorus application is sufficient.',
        'economic_value': 'Key ingredient in snack food manufacturing (Bikaneri bhujia) and protein security in drylands.'
    },
    'mungbean': {
        'display_name': 'Mung Bean (Green Gram)',
        'scientific_name': 'Vigna radiata',
        'category': 'Pulse / Short-Duration',
        'icon': '🟢',
        'tagline': 'Short-cycle quick cash pulse fitting tight crop rotations.',
        'description': 'Mung bean is a rapid-maturing warm-season legume that matures in around 60-70 days. Excellent as an intercrop and green manure.',
        'soil_type': 'Fertile loam to sandy loam with good drainage.',
        'season': 'Kharif / Summer (Zaid)',
        'growth_duration': '60 - 75 days',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '35 - 60 kg/ha', 'K': '15 - 25 kg/ha', 'temp': '27 - 30 °C', 'humidity': '80 - 90 %', 'ph': '6.2 - 7.2', 'rainfall': '35 - 60 mm', 'soil_moisture': '38 - 56 %', 'sunlight_hours': '7.0 - 8.8 hrs/day'},
        'irrigation_tips': '2-3 light irrigations are sufficient. Stop irrigation 10 days before harvest.',
        'fertilizer_tips': 'Basal dose of DAP or single superphosphate (SSP) ensures vigorous nodulation.',
        'economic_value': 'Quick returns in just 2 months, high market price, restores nitrogen to soil.'
    },
    'blackgram': {
        'display_name': 'Black Gram (Urad)',
        'scientific_name': 'Vigna mungo',
        'category': 'Pulse / Legume',
        'icon': '⚫',
        'tagline': 'High phosphoric nutrient consumer producing protein-rich urad.',
        'description': 'Black gram is a prominent warm-season pulse widely used in culinary staples like idlis, dosas, and dal makhani. It prefers warm, humid climates.',
        'soil_type': 'Heavy clay, alluvial, or loamy soil with moderate moisture retention.',
        'season': 'Kharif / Spring',
        'growth_duration': '75 - 90 days',
        'optimal_ranges': {'N': '30 - 60 kg/ha', 'P': '55 - 80 kg/ha', 'K': '15 - 25 kg/ha', 'temp': '25 - 35 °C', 'humidity': '60 - 70 %', 'ph': '6.5 - 7.8', 'rainfall': '60 - 75 mm', 'soil_moisture': '40 - 58 %', 'sunlight_hours': '7.0 - 8.8 hrs/day'},
        'irrigation_tips': 'Requires consistent moisture during vegetative stage; sensitive to standing water.',
        'fertilizer_tips': 'Apply balanced NPK along with Sulfur (20 kg/ha) to enhance protein synthesis.',
        'economic_value': 'High commercial value across South Asian cuisine, dependable cash crop.'
    },
    'lentil': {
        'display_name': 'Lentil (Masoor)',
        'scientific_name': 'Lens culinaris',
        'category': 'Pulse / Legume',
        'icon': '🥣',
        'tagline': 'Cool season nutritious pulse with low water requirement.',
        'description': 'Lentils are ancient cool-season legumes celebrated for their high iron and protein content. They grow well in low moisture conditions with minimal fertilizer.',
        'soil_type': 'Light loams, alluvial, and well-drained clay loams.',
        'season': 'Rabi (Winter)',
        'growth_duration': '100 - 130 days',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '55 - 80 kg/ha', 'K': '15 - 25 kg/ha', 'temp': '18 - 30 °C', 'humidity': '60 - 70 %', 'ph': '6.0 - 7.5', 'rainfall': '35 - 55 mm', 'soil_moisture': '24 - 40 %', 'sunlight_hours': '6.5 - 8.5 hrs/day'},
        'irrigation_tips': 'Requires 1-2 light irrigations if winter rains fail. Avoid excess water.',
        'fertilizer_tips': 'Phosphorus and biofertilizers (Rhizobium + PSB) maximize pod set.',
        'economic_value': 'Global export potential, low input cost, high profit margins.'
    },
    'pomegranate': {
        'display_name': 'Pomegranate',
        'scientific_name': 'Punica granatum',
        'category': 'Horticultural Fruit',
        'icon': '🍎',
        'tagline': 'Semi-arid fruit crop with high antioxidant commercial value.',
        'description': 'Pomegranate is an evergreen or deciduous fruit shrub renowned for antioxidant-rich arils. Highly tolerant to drought, semi-arid conditions, and salinity.',
        'soil_type': 'Deep, well-drained loamy to sandy loam soils.',
        'season': 'Perennial (Year-round fruiting seasons)',
        'growth_duration': '150 - 180 days (flowering to harvest)',
        'optimal_ranges': {'N': '15 - 40 kg/ha', 'P': '10 - 25 kg/ha', 'K': '35 - 45 kg/ha', 'temp': '18 - 25 °C', 'humidity': '85 - 95 %', 'ph': '5.5 - 7.2', 'rainfall': '100 - 115 mm', 'soil_moisture': '35 - 55 %', 'sunlight_hours': '7.5 - 9.8 hrs/day'},
        'irrigation_tips': 'Drip irrigation is ideal. Regulate watering during fruit ripening to prevent cracking.',
        'fertilizer_tips': 'High Potassium and Boron requirements during fruit development to ensure deep red coloration.',
        'economic_value': 'Super premium export fruit, massive pharmaceutical and juice industry demand.'
    },
    'banana': {
        'display_name': 'Banana',
        'scientific_name': 'Musa acuminata',
        'category': 'Plantation Fruit',
        'icon': '🍌',
        'tagline': 'Nutrient-heavy tropical giant with high yield potential.',
        'description': 'Banana is one of the highest-yielding tropical fruit crops. It requires copious amounts of nitrogen, potassium, warm temperatures, and abundant water.',
        'soil_type': 'Deep, fertile alluvial and volcanic loam rich in organic matter.',
        'season': 'Year-round planting in tropical zones',
        'growth_duration': '10 - 12 months',
        'optimal_ranges': {'N': '90 - 120 kg/ha', 'P': '70 - 95 kg/ha', 'K': '45 - 55 kg/ha', 'temp': '25 - 30 °C', 'humidity': '75 - 85 %', 'ph': '5.5 - 6.5', 'rainfall': '90 - 120 mm', 'soil_moisture': '65 - 85 %', 'sunlight_hours': '6.5 - 8.5 hrs/day'},
        'irrigation_tips': 'Continuous soil moisture is critical; drip fertigation yields 30-40% higher bunches.',
        'fertilizer_tips': 'Heaviest consumer of Potassium and Nitrogen. Apply fertilizers in regular monthly doses.',
        'economic_value': 'Massive domestic consumption, continuous revenue cycles, reliable profit.'
    },
    'mango': {
        'display_name': 'Mango',
        'scientific_name': 'Mangifera indica',
        'category': 'Tropical Fruit',
        'icon': '🥭',
        'tagline': 'The King of Fruits thriving in tropical sunny plains.',
        'description': 'Mango is the premier tropical fruit of South Asia. It requires a distinct dry spell for flowering and warm sunny weather during fruit maturation.',
        'soil_type': 'Deep, well-drained alluvial or red loamy soil with depth > 2 meters.',
        'season': 'Summer harvest (Perennial tree)',
        'growth_duration': '100 - 130 days (from bloom to maturity)',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '15 - 40 kg/ha', 'K': '25 - 35 kg/ha', 'temp': '27 - 36 °C', 'humidity': '45 - 55 %', 'ph': '4.5 - 7.0', 'rainfall': '85 - 105 mm', 'soil_moisture': '35 - 55 %', 'sunlight_hours': '8.2 - 10.5 hrs/day'},
        'irrigation_tips': 'Withhold irrigation 2 months prior to flowering. Resume irrigation once fruits reach pea size.',
        'fertilizer_tips': 'Apply farmyard manure, Nitrogen, and Potassium after annual post-harvest pruning.',
        'economic_value': 'Very high export value, processing into pulp, juices, and specialty preserves.'
    },
    'grapes': {
        'display_name': 'Grapes',
        'scientific_name': 'Vitis vinifera',
        'category': 'Viticulture / Fruit',
        'icon': '🍇',
        'tagline': 'High potassium consumer yielding premium table and wine fruit.',
        'description': 'Grapes require temperate or subtropical climates with dry summers. They thrive in mineral-rich soils and require specialized trellising and canopy management.',
        'soil_type': 'Well-drained sandy loam or gravelly soil with good permeability.',
        'season': 'Winter pruning to Summer harvest',
        'growth_duration': '120 - 150 days (pruning to harvest)',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '120 - 145 kg/ha', 'K': '195 - 205 kg/ha', 'temp': '10 - 40 °C', 'humidity': '80 - 85 %', 'ph': '5.5 - 6.5', 'rainfall': '65 - 75 mm', 'soil_moisture': '42 - 60 %', 'sunlight_hours': '7.5 - 9.5 hrs/day'},
        'irrigation_tips': 'Sub-surface drip irrigation is standard. Water stress during veraison improves sugar concentration.',
        'fertilizer_tips': 'Very high Potassium and Phosphorus requirements. Supplement Magnesium and micronutrients.',
        'economic_value': 'High return per acre, export table grapes, wine production, and raisin processing.'
    },
    'watermelon': {
        'display_name': 'Watermelon',
        'scientific_name': 'Citrullus lanatus',
        'category': 'Cucurbit Fruit',
        'icon': '🍉',
        'tagline': 'Fast-growing summer refresher loving sandy warm beds.',
        'description': 'Watermelon is a warm-season annual vine producing large, juicy fruits. It requires long sunny days, warm nights, and dry weather during fruit ripening.',
        'soil_type': 'Deep sandy loam with rich organic matter and rapid drainage.',
        'season': 'Summer (Zaid season)',
        'growth_duration': '80 - 100 days',
        'optimal_ranges': {'N': '80 - 120 kg/ha', 'P': '10 - 30 kg/ha', 'K': '45 - 55 kg/ha', 'temp': '24 - 27 °C', 'humidity': '80 - 90 %', 'ph': '6.0 - 7.0', 'rainfall': '40 - 60 mm', 'soil_moisture': '32 - 48 %', 'sunlight_hours': '8.8 - 10.8 hrs/day'},
        'irrigation_tips': 'Frequent light irrigations during vine growth; reduce watering near harvest to prevent fruit bursting.',
        'fertilizer_tips': 'Nitrogen for foliage expansion, followed by high Potassium and Calcium for crisp flesh.',
        'economic_value': 'Short-duration summer crop providing rapid cash return and high profit margins.'
    },
    'muskmelon': {
        'display_name': 'Muskmelon (Cantaloupe)',
        'scientific_name': 'Cucumis melo',
        'category': 'Cucurbit Fruit',
        'icon': '🍈',
        'tagline': 'Sweet aromatic summer fruit requiring warm dry atmosphere.',
        'description': 'Muskmelon thrives in sunny, warm climates with low humidity during maturity to maximize sugar sweetness and aromatic flavor.',
        'soil_type': 'Sandy loam, alluvial riverbeds with rapid drainage.',
        'season': 'Summer (Zaid)',
        'growth_duration': '75 - 90 days',
        'optimal_ranges': {'N': '80 - 120 kg/ha', 'P': '10 - 30 kg/ha', 'K': '45 - 55 kg/ha', 'temp': '27 - 30 °C', 'humidity': '90 - 95 %', 'ph': '6.0 - 6.8', 'rainfall': '20 - 30 mm', 'soil_moisture': '30 - 45 %', 'sunlight_hours': '8.5 - 10.5 hrs/day'},
        'irrigation_tips': 'Drip irrigation avoids wetting vines and reduces downy mildew risk.',
        'fertilizer_tips': 'Potassium silicate enhances rind thickness and shelf life during transit.',
        'economic_value': 'High urban retail demand, quick crop cycle, excellent intercropping option.'
    },
    'apple': {
        'display_name': 'Apple',
        'scientific_name': 'Malus domestica',
        'category': 'Temperate Fruit',
        'icon': '🍏',
        'tagline': 'Cool highland fruit demanding chilling hours and high potassium.',
        'description': 'Apple is the premier temperate fruit requiring 1000-1500 chilling hours below 7°C for bud dormancy break, high altitude conditions, and fertile loamy soil.',
        'soil_type': 'Deep, well-aerated loamy soil with organic humus and slope drainage.',
        'season': 'Spring bloom to Autumn harvest',
        'growth_duration': '130 - 160 days (bloom to maturity)',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '120 - 145 kg/ha', 'K': '195 - 205 kg/ha', 'temp': '21 - 24 °C', 'humidity': '90 - 95 %', 'ph': '5.5 - 6.5', 'rainfall': '100 - 125 mm', 'soil_moisture': '50 - 70 %', 'sunlight_hours': '5.0 - 7.0 hrs/day'},
        'irrigation_tips': 'Consistent moisture during fruit development; drip irrigation prevents fungal leaf diseases.',
        'fertilizer_tips': 'High Phosphorus and Potassium ensure crisp cell structure and deep skin coloration.',
        'economic_value': 'Premium long-term orchard investment, cold-storage capability, high retail prices.'
    },
    'orange': {
        'display_name': 'Orange (Citrus)',
        'scientific_name': 'Citrus sinensis',
        'category': 'Citrus Fruit',
        'icon': '🍊',
        'tagline': 'Subtropical citrus rich in Vitamin C and citrus oils.',
        'description': 'Oranges grow best in subtropical climates without severe frost. They require ample sunlight, good drainage, and balanced micronutrients.',
        'soil_type': 'Well-drained light loamy and sandy loam soils with permeable subsoil.',
        'season': 'Winter to Early Spring harvest',
        'growth_duration': '240 - 280 days (flowering to harvest)',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '10 - 30 kg/ha', 'K': '5 - 15 kg/ha', 'temp': '10 - 35 °C', 'humidity': '90 - 95 %', 'ph': '6.0 - 8.0', 'rainfall': '100 - 120 mm', 'soil_moisture': '48 - 68 %', 'sunlight_hours': '6.5 - 8.5 hrs/day'},
        'irrigation_tips': 'Avoid water standing around tree trunks to prevent gummosis; drip rings are ideal.',
        'fertilizer_tips': 'Foliar spray of Zinc, Iron, and Manganese corrects frequent citrus mottling.',
        'economic_value': 'Steady high market demand, fresh fruit juice manufacturing, long shelf life.'
    },
    'papaya': {
        'display_name': 'Papaya',
        'scientific_name': 'Carica papaya',
        'category': 'Tropical Fruit',
        'icon': '🥭',
        'tagline': 'Fast-growing tropical fruit yielding heavy harvest within a year.',
        'description': 'Papaya is a fast-growing, heavy-yielding herbaceous fruit tree that produces year-round in frost-free tropical and subtropical areas.',
        'soil_type': 'Rich alluvial loam, volcanic soil with quick drainage (highly intolerant to waterlogging).',
        'season': 'Year-round planting and harvest',
        'growth_duration': '9 - 11 months (to first harvest)',
        'optimal_ranges': {'N': '30 - 70 kg/ha', 'P': '45 - 70 kg/ha', 'K': '45 - 55 kg/ha', 'temp': '23 - 44 °C', 'humidity': '90 - 95 %', 'ph': '6.0 - 7.0', 'rainfall': '140 - 250 mm', 'soil_moisture': '58 - 78 %', 'sunlight_hours': '7.0 - 9.0 hrs/day'},
        'irrigation_tips': 'Light, frequent watering. Never allow water to stand around base of plant.',
        'fertilizer_tips': 'Regular monthly doses of NPK and Boron prevent fruit deformation.',
        'economic_value': 'Rapid commercial return, valuable enzyme (papain) extraction, continuous weekly yield.'
    },
    'coconut': {
        'display_name': 'Coconut',
        'scientific_name': 'Cocos nucifera',
        'category': 'Plantation Palm',
        'icon': '🥥',
        'tagline': 'Tree of life thriving in coastal humidity and tropical sun.',
        'description': 'Coconut is known as the "Tree of Life" because every part of the palm is commercially useful. It flourishes in tropical coastal belts with high humidity.',
        'soil_type': 'Coastal sands, alluvial loams, and lateritic soils with good drainage.',
        'season': 'Perennial (Year-round continuous monthly harvests)',
        'growth_duration': '12 months per bunch cycle',
        'optimal_ranges': {'N': '10 - 40 kg/ha', 'P': '10 - 30 kg/ha', 'K': '25 - 35 kg/ha', 'temp': '25 - 29 °C', 'humidity': '95 - 100 %', 'ph': '5.5 - 6.5', 'rainfall': '130 - 225 mm', 'soil_moisture': '62 - 82 %', 'sunlight_hours': '6.8 - 8.8 hrs/day'},
        'irrigation_tips': 'Requires 40-50 liters of water per palm per day during summer months.',
        'fertilizer_tips': 'Requires substantial Potassium and Sodium (common salt) for higher nut yield.',
        'economic_value': 'High value from tender water, copra, coconut oil, coir fiber, and shell charcoal.'
    },
    'cotton': {
        'display_name': 'Cotton',
        'scientific_name': 'Gossypium hirsutum',
        'category': 'Fiber / Cash Crop',
        'icon': '🧵',
        'tagline': 'White Gold - Premier natural fiber crop of black clay soils.',
        'description': 'Cotton is the world\'s leading natural textile fiber. It requires a long frost-free period, high sunshine, and deep black soils with good moisture retention.',
        'soil_type': 'Deep black cotton soil (Regur) or fertile alluvial loam.',
        'season': 'Kharif',
        'growth_duration': '150 - 180 days',
        'optimal_ranges': {'N': '100 - 140 kg/ha', 'P': '35 - 60 kg/ha', 'K': '15 - 25 kg/ha', 'temp': '22 - 26 °C', 'humidity': '75 - 85 %', 'ph': '6.0 - 8.0', 'rainfall': '60 - 100 mm', 'soil_moisture': '40 - 58 %', 'sunlight_hours': '8.2 - 10.2 hrs/day'},
        'irrigation_tips': 'Drip irrigation ensures uniform boll bursting. Avoid excess water during boll opening.',
        'fertilizer_tips': 'Apply split Nitrogen along with Magnesium and Boron to prevent leaf reddening and square dropping.',
        'economic_value': 'High value export commodity, backbone of global textile and ginning industry.'
    },
    'jute': {
        'display_name': 'Jute',
        'scientific_name': 'Corchorus capsularis',
        'category': 'Fiber / Golden Fiber',
        'icon': '🌾',
        'tagline': 'Golden Fiber thriving in humid deltaic monsoon floodplains.',
        'description': 'Jute is the 100% biodegradable "Golden Fiber". It requires high temperatures, heavy monsoon precipitation, and river basin alluvial silts.',
        'soil_type': 'New alluvial silt loam in river delta plains.',
        'season': 'Kharif / Pre-monsoon',
        'growth_duration': '120 - 150 days',
        'optimal_ranges': {'N': '60 - 100 kg/ha', 'P': '35 - 60 kg/ha', 'K': '35 - 45 kg/ha', 'temp': '23 - 26 °C', 'humidity': '70 - 90 %', 'ph': '6.0 - 7.5', 'rainfall': '150 - 200 mm', 'soil_moisture': '72 - 92 %', 'sunlight_hours': '5.5 - 7.5 hrs/day'},
        'irrigation_tips': 'Requires moist soil at germination followed by plentiful rainfall or standing water during late growth.',
        'fertilizer_tips': 'Apply Nitrogen top-dressings at 3-4 weeks to stimulate tall, unbranched stem height.',
        'economic_value': 'Eco-friendly alternative to plastic packaging, sacks, geotextiles, and handicrafts.'
    },
    'coffee': {
        'display_name': 'Coffee',
        'scientific_name': 'Coffea arabica',
        'category': 'Plantation Beverage',
        'icon': '☕',
        'tagline': 'High-altitude shade-grown beverage crop with rich aroma.',
        'description': 'Coffee thrives on tropical hill slopes under two-tier shade trees. It demands rich volcanic or forest humus soil, moderate rainfall, and cool misty air.',
        'soil_type': 'Deep, porous, rich volcanic loam or forest humus with acidic to neutral pH.',
        'season': 'Autumn to Winter picking',
        'growth_duration': '210 - 240 days (flowering to cherry ripening)',
        'optimal_ranges': {'N': '80 - 120 kg/ha', 'P': '15 - 40 kg/ha', 'K': '25 - 35 kg/ha', 'temp': '23 - 28 °C', 'humidity': '50 - 70 %', 'ph': '6.0 - 7.5', 'rainfall': '115 - 200 mm', 'soil_moisture': '55 - 75 %', 'sunlight_hours': '5.0 - 6.8 hrs/day'},
        'irrigation_tips': 'Blossom showers in March-April trigger uniform flowering and high berry set.',
        'fertilizer_tips': 'Apply balanced NPK enriched with organic compost and zinc during pre-monsoon and post-monsoon.',
        'economic_value': 'Global luxury commodity, specialty coffee export premiums, agro-tourism appeal.'
    }
}

# Empirical average benchmarks for 9 features across all 24 crops
PARAM_STATS = {
    'apple': {'N': 20.8, 'P': 134.2, 'K': 199.9, 'temp': 22.6, 'humidity': 92.3, 'ph': 5.9, 'rainfall': 112.7, 'moisture': 59.0, 'sunlight': 5.7},
    'banana': {'N': 100.2, 'P': 82.0, 'K': 50.0, 'temp': 27.4, 'humidity': 80.4, 'ph': 6.0, 'rainfall': 104.6, 'moisture': 74.2, 'sunlight': 7.3},
    'blackgram': {'N': 40.0, 'P': 67.5, 'K': 19.2, 'temp': 30.0, 'humidity': 65.1, 'ph': 7.1, 'rainfall': 67.9, 'moisture': 49.4, 'sunlight': 7.8},
    'chickpea': {'N': 40.1, 'P': 67.8, 'K': 79.9, 'temp': 18.9, 'humidity': 16.9, 'ph': 7.3, 'rainfall': 80.1, 'moisture': 27.2, 'sunlight': 8.5},
    'coconut': {'N': 22.0, 'P': 16.9, 'K': 30.6, 'temp': 27.4, 'humidity': 94.8, 'ph': 6.0, 'rainfall': 175.7, 'moisture': 74.1, 'sunlight': 7.4},
    'coffee': {'N': 101.2, 'P': 28.7, 'K': 29.9, 'temp': 25.5, 'humidity': 58.9, 'ph': 6.8, 'rainfall': 158.1, 'moisture': 66.1, 'sunlight': 5.9},
    'cotton': {'N': 117.8, 'P': 46.2, 'K': 19.6, 'temp': 24.0, 'humidity': 79.8, 'ph': 6.9, 'rainfall': 80.4, 'moisture': 47.4, 'sunlight': 8.9},
    'grapes': {'N': 23.2, 'P': 132.5, 'K': 200.1, 'temp': 23.8, 'humidity': 81.9, 'ph': 6.0, 'rainfall': 69.6, 'moisture': 51.5, 'sunlight': 8.3},
    'jute': {'N': 78.4, 'P': 46.9, 'K': 40.0, 'temp': 25.0, 'humidity': 79.6, 'ph': 6.7, 'rainfall': 174.8, 'moisture': 82.1, 'sunlight': 6.5},
    'kidneybeans': {'N': 20.8, 'P': 67.5, 'K': 20.1, 'temp': 20.1, 'humidity': 21.6, 'ph': 5.7, 'rainfall': 105.9, 'moisture': 41.8, 'sunlight': 7.8},
    'lentil': {'N': 18.8, 'P': 68.4, 'K': 19.4, 'temp': 24.5, 'humidity': 64.8, 'ph': 6.9, 'rainfall': 45.7, 'moisture': 30.5, 'sunlight': 7.4},
    'maize': {'N': 77.8, 'P': 48.4, 'K': 19.8, 'temp': 22.4, 'humidity': 65.1, 'ph': 6.2, 'rainfall': 84.8, 'moisture': 55.3, 'sunlight': 8.6},
    'mango': {'N': 20.1, 'P': 27.2, 'K': 29.9, 'temp': 31.2, 'humidity': 50.2, 'ph': 5.8, 'rainfall': 94.7, 'moisture': 44.5, 'sunlight': 9.4},
    'mothbeans': {'N': 21.4, 'P': 48.0, 'K': 20.2, 'temp': 28.2, 'humidity': 53.2, 'ph': 6.8, 'rainfall': 51.2, 'moisture': 19.5, 'sunlight': 9.5},
    'mungbean': {'N': 21.0, 'P': 47.3, 'K': 19.9, 'temp': 28.5, 'humidity': 85.5, 'ph': 6.7, 'rainfall': 48.4, 'moisture': 46.3, 'sunlight': 7.7},
    'muskmelon': {'N': 100.3, 'P': 17.7, 'K': 50.1, 'temp': 28.7, 'humidity': 92.3, 'ph': 6.4, 'rainfall': 24.7, 'moisture': 35.6, 'sunlight': 9.1},
    'orange': {'N': 19.6, 'P': 16.6, 'K': 10.0, 'temp': 22.8, 'humidity': 92.2, 'ph': 7.0, 'rainfall': 110.5, 'moisture': 58.6, 'sunlight': 7.4},
    'papaya': {'N': 49.9, 'P': 59.0, 'K': 50.0, 'temp': 33.7, 'humidity': 92.4, 'ph': 6.7, 'rainfall': 142.6, 'moisture': 70.3, 'sunlight': 7.7},
    'pigeonpeas': {'N': 20.7, 'P': 67.7, 'K': 20.3, 'temp': 27.7, 'humidity': 48.1, 'ph': 5.8, 'rainfall': 149.5, 'moisture': 37.2, 'sunlight': 8.3},
    'pomegranate': {'N': 18.9, 'P': 18.8, 'K': 40.2, 'temp': 21.8, 'humidity': 90.1, 'ph': 6.4, 'rainfall': 107.5, 'moisture': 45.1, 'sunlight': 8.4},
    'rice': {'N': 79.9, 'P': 47.6, 'K': 39.9, 'temp': 23.7, 'humidity': 82.3, 'ph': 6.4, 'rainfall': 236.2, 'moisture': 85.5, 'sunlight': 6.3},
    'sugarcane': {'N': 145.1, 'P': 66.9, 'K': 91.8, 'temp': 30.5, 'humidity': 79.3, 'ph': 6.8, 'rainfall': 197.1, 'moisture': 78.6, 'sunlight': 8.8},
    'watermelon': {'N': 99.4, 'P': 17.0, 'K': 50.2, 'temp': 25.6, 'humidity': 85.2, 'ph': 6.5, 'rainfall': 50.8, 'moisture': 39.0, 'sunlight': 9.5},
    'wheat': {'N': 103.8, 'P': 54.8, 'K': 39.3, 'temp': 18.3, 'humidity': 57.6, 'ph': 6.7, 'rainfall': 70.4, 'moisture': 51.8, 'sunlight': 8.1}
}

# -------------------------------------------------------------
# 3. HELPER FUNCTIONS (9 INPUT VALIDATION)
# -------------------------------------------------------------
def validate_inputs(form_data):
    """Validates 9 user agricultural inputs and converts them to float."""
    errors = []
    cleaned = {}
    
    limits = {
        'N': {'name': 'Nitrogen (N)', 'min': 0.0, 'max': 200.0, 'unit': 'kg/ha'},
        'P': {'name': 'Phosphorus (P)', 'min': 0.0, 'max': 200.0, 'unit': 'kg/ha'},
        'K': {'name': 'Potassium (K)', 'min': 0.0, 'max': 250.0, 'unit': 'kg/ha'},
        'temperature': {'name': 'Temperature', 'min': 0.0, 'max': 55.0, 'unit': '°C'},
        'humidity': {'name': 'Humidity', 'min': 5.0, 'max': 100.0, 'unit': '%'},
        'ph': {'name': 'Soil pH', 'min': 0.0, 'max': 14.0, 'unit': 'scale (0-14)'},
        'rainfall': {'name': 'Rainfall', 'min': 0.0, 'max': 500.0, 'unit': 'mm'},
        'soil_moisture': {'name': 'Soil Moisture', 'min': 0.0, 'max': 100.0, 'unit': '%'},
        'sunlight_hours': {'name': 'Sunlight Duration', 'min': 0.0, 'max': 16.0, 'unit': 'hrs/day'}
    }
    
    for key, spec in limits.items():
        val_str = form_data.get(key)
        if val_str is None or str(val_str).strip() == '':
            # Fallback default if optional from simple request
            if key == 'soil_moisture':
                val_str = '50.0'
            elif key == 'sunlight_hours':
                val_str = '8.0'
            else:
                errors.append(f"Please provide a value for {spec['name']}.")
                continue
        try:
            val = float(val_str)
            if val < spec['min'] or val > spec['max']:
                errors.append(f"{spec['name']} value ({val}) is out of realistic agricultural range [{spec['min']} - {spec['max']} {spec['unit']}].")
            cleaned[key] = val
        except ValueError:
            errors.append(f"Invalid numeric input for {spec['name']}.")
            
    return cleaned, errors

def predict_crop_with_ml(input_data):
    """Runs 9-feature ML inference, calculates probabilities and alternative predictions."""
    if model_bundle is None:
        load_ml_model()
        if model_bundle is None:
            raise RuntimeError("Machine learning model is not available. Please train the model.")
            
    rf = model_bundle['model']
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_moisture', 'sunlight_hours']
    
    df_input = pd.DataFrame([[input_data[f] for f in features]], columns=features)
    
    # Primary Prediction
    predicted_label = rf.predict(df_input)[0]
    
    # Class Probabilities
    probabilities = rf.predict_proba(df_input)[0]
    classes = rf.classes_
    
    ranked_indices = np.argsort(probabilities)[::-1]
    confidence = float(probabilities[ranked_indices[0]] * 100)
    
    top_alternatives = []
    for idx in ranked_indices[1:4]:
        prob = float(probabilities[idx] * 100)
        crop_key = classes[idx]
        if prob > 0.5:
            top_alternatives.append({
                'crop_key': crop_key,
                'name': CROP_DETAILS.get(crop_key, {}).get('display_name', crop_key.title()),
                'icon': CROP_DETAILS.get(crop_key, {}).get('icon', '🌱'),
                'probability': round(prob, 1),
                'category': CROP_DETAILS.get(crop_key, {}).get('category', 'Crop')
            })
            
    crop_info = CROP_DETAILS.get(predicted_label, {})
    param_analysis = []
    ideal_benchmarks = PARAM_STATS.get(predicted_label, {})
    
    param_display_names = {
        'N': 'Nitrogen (N)',
        'P': 'Phosphorus (P)',
        'K': 'Potassium (K)',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'ph': 'Soil pH',
        'rainfall': 'Rainfall',
        'soil_moisture': 'Soil Moisture',
        'sunlight_hours': 'Sunlight Duration'
    }
    param_units = {
        'N': 'kg/ha',
        'P': 'kg/ha',
        'K': 'kg/ha',
        'temperature': '°C',
        'humidity': '%',
        'ph': '',
        'rainfall': 'mm',
        'soil_moisture': '%',
        'sunlight_hours': 'hrs/day'
    }
    
    for f in features:
        user_val = input_data[f]
        stat_key = 'temp' if f == 'temperature' else ('moisture' if f == 'soil_moisture' else ('sunlight' if f == 'sunlight_hours' else f))
        ideal_val = ideal_benchmarks.get(stat_key, user_val)
        
        diff = abs(user_val - ideal_val)
        pct_diff = (diff / (ideal_val + 1e-5)) * 100
        
        if pct_diff <= 25:
            status = 'Optimal'
            badge_class = 'badge-optimal'
        elif pct_diff <= 50:
            status = 'Moderate'
            badge_class = 'badge-moderate'
        else:
            status = 'Suboptimal'
            badge_class = 'badge-suboptimal'
            
        param_analysis.append({
            'name': param_display_names[f],
            'user_val': user_val,
            'ideal_val': ideal_val,
            'unit': param_units[f],
            'status': status,
            'badge_class': badge_class
        })
        
    return {
        'predicted_crop': predicted_label,
        'confidence': round(confidence, 1),
        'crop_info': crop_info,
        'alternatives': top_alternatives,
        'param_analysis': param_analysis,
        'user_inputs': input_data
    }

# -------------------------------------------------------------
# 4. WEB APPLICATION ROUTES
# -------------------------------------------------------------

@app.route('/')
def index():
    """Renders Homepage with 9-Feature Input Form, Presets, and 24-Crop Library."""
    presets = [
        # Row 1 (5 Scenarios)
        {'name': 'Rice Field', 'icon': '🌾', 'N': 90, 'P': 42, 'K': 43, 'temp': 20.9, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9, 'moisture': 85.0, 'sunlight': 6.5, 'badge': 'High Water'},
        {'name': 'Wheat Farm', 'icon': '🍞', 'N': 105, 'P': 55, 'K': 40, 'temp': 18.5, 'humidity': 58.0, 'ph': 6.7, 'rainfall': 70.0, 'moisture': 52.0, 'sunlight': 8.0, 'badge': 'Rabi Winter'},
        {'name': 'Sugarcane Estate', 'icon': '🎋', 'N': 148, 'P': 68, 'K': 92, 'temp': 30.5, 'humidity': 80.0, 'ph': 6.9, 'rainfall': 195.0, 'moisture': 78.0, 'sunlight': 8.8, 'badge': 'High Yield'},
        {'name': 'Maize Farm', 'icon': '🌽', 'N': 80, 'P': 50, 'K': 20, 'temp': 22.5, 'humidity': 65.0, 'ph': 6.2, 'rainfall': 85.0, 'moisture': 55.0, 'sunlight': 8.5, 'badge': 'Moderate'},
        {'name': 'Chickpea Field', 'icon': '🧆', 'N': 40, 'P': 68, 'K': 80, 'temp': 18.9, 'humidity': 17.0, 'ph': 7.3, 'rainfall': 80.0, 'moisture': 28.0, 'sunlight': 8.5, 'badge': 'Dry Pulse'},
        # Row 2 (5 Scenarios)
        {'name': 'Coffee Estate', 'icon': '☕', 'N': 105, 'P': 25, 'K': 30, 'temp': 25.5, 'humidity': 58.0, 'ph': 6.8, 'rainfall': 160.0, 'moisture': 65.0, 'sunlight': 5.8, 'badge': 'Highland Shade'},
        {'name': 'Apple Orchard', 'icon': '🍏', 'N': 20, 'P': 135, 'K': 200, 'temp': 22.5, 'humidity': 92.0, 'ph': 6.0, 'rainfall': 110.0, 'moisture': 60.0, 'sunlight': 6.2, 'badge': 'High Potash'},
        {'name': 'Cotton Fields', 'icon': '🧵', 'N': 120, 'P': 45, 'K': 20, 'temp': 24.5, 'humidity': 80.0, 'ph': 6.9, 'rainfall': 80.0, 'moisture': 48.0, 'sunlight': 9.2, 'badge': 'Black Soil'},
        {'name': 'Banana Grove', 'icon': '🍌', 'N': 100, 'P': 82, 'K': 50, 'temp': 27.4, 'humidity': 80.4, 'ph': 6.0, 'rainfall': 105.0, 'moisture': 74.0, 'sunlight': 7.3, 'badge': 'Tropical Fruit'},
        {'name': 'Watermelon Patch', 'icon': '🍉', 'N': 100, 'P': 18, 'K': 50, 'temp': 26.0, 'humidity': 85.0, 'ph': 6.5, 'rainfall': 50.0, 'moisture': 40.0, 'sunlight': 9.8, 'badge': 'Summer Zaid'}
    ]
    return render_template('index.html', presets=presets, crops=CROP_DETAILS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles HTML form submission for crop recommendation."""
    cleaned_inputs, errors = validate_inputs(request.form)
    
    if errors:
        for err in errors:
            flash(err, 'error')
        return redirect(url_for('index'))
        
    try:
        results = predict_crop_with_ml(cleaned_inputs)
        return render_template('result.html', result=results)
    except Exception as e:
        flash(f"Prediction Error: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for real-time AJAX prediction."""
    data = request.get_json(silent=True) or request.form
    cleaned_inputs, errors = validate_inputs(data)
    
    if errors:
        return jsonify({'success': False, 'errors': errors}), 400
        
    try:
        results = predict_crop_with_ml(cleaned_inputs)
        return jsonify({
            'success': True,
            'prediction': results['predicted_crop'],
            'confidence': results['confidence'],
            'crop_info': results['crop_info'],
            'alternatives': results['alternatives'],
            'param_analysis': results['param_analysis']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/crop/<crop_name>')
def crop_api(crop_name):
    """Returns specific crop agronomic data."""
    crop_name = crop_name.lower().strip()
    if crop_name in CROP_DETAILS:
        return jsonify({'success': True, 'crop': CROP_DETAILS[crop_name]})
    return jsonify({'success': False, 'error': 'Crop not found'}), 404

# -------------------------------------------------------------
# 5. ENTRY POINT
# -------------------------------------------------------------
if __name__ == '__main__':
    print("[*] Starting Crop Recommendation System Flask Server (9 Features & 24 Crops)...")
    print("[*] Open http://127.0.0.1:5000 in your web browser.")
    app.run(debug=True, host='0.0.0.0', port=5000)
