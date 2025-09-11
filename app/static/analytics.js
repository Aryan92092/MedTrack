/**
 * Advanced Analytics and Data Visualization Module
 */

class AnalyticsDashboard {
    constructor() {
        this.charts = {};
        this.data = {};
        this.init();
    }

    init() {
        this.loadAnalyticsData();
        this.setupEventListeners();
    }

    async loadAnalyticsData() {
        try {
            // Load consumption trends
            const consumptionResponse = await fetch('/analytics/api/consumption-trends?days=30');
            this.data.consumption = await consumptionResponse.json();

            // Load ABC analysis
            const abcResponse = await fetch('/analytics/api/abc-analysis');
            this.data.abc = await abcResponse.json();

            // Load XYZ analysis
            const xyzResponse = await fetch('/analytics/api/xyz-analysis');
            this.data.xyz = await xyzResponse.json();

            // Load recommendations
            const recommendationsResponse = await fetch('/analytics/api/recommendations');
            this.data.recommendations = await recommendationsResponse.json();

            this.renderCharts();
            this.renderRecommendations();
        } catch (error) {
            console.error('Error loading analytics data:', error);
        }
    }

    setupEventListeners() {
        // Chart refresh buttons
        document.querySelectorAll('.refresh-chart').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const chartType = e.target.dataset.chart;
                this.refreshChart(chartType);
            });
        });

        // Date range selectors
        document.querySelectorAll('.date-range-selector').forEach(selector => {
            selector.addEventListener('change', (e) => {
                this.updateDateRange(e.target.value);
            });
        });
    }

    renderCharts() {
        this.renderConsumptionChart();
        this.renderCategoryChart();
        this.renderRiskChart();
        this.renderStockChart();
    }

    renderConsumptionChart() {
        const ctx = document.getElementById('consumptionChart');
        if (!ctx) return;

        const data = this.data.consumption.daily_trends || {};
        const labels = Object.keys(data).sort();
        const values = labels.map(label => data[label] || 0);

        this.charts.consumption = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Daily Consumption',
                    data: values,
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Consumption Trends (30 days)'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Quantity Consumed'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    renderCategoryChart() {
        const ctx = document.getElementById('categoryChart');
        if (!ctx) return;

        // This would need to be passed from the backend
        const categories = {
            'Pain Relief': 15,
            'Antibiotics': 8,
            'Vitamins': 12,
            'Cough & Cold': 6,
            'Other': 4
        };

        const labels = Object.keys(categories);
        const values = Object.values(categories);
        const colors = [
            'rgba(34, 197, 94, 0.8)',
            'rgba(59, 130, 246, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(239, 68, 68, 0.8)',
            'rgba(139, 92, 246, 0.8)'
        ];

        this.charts.category = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors.slice(0, labels.length),
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Medicine Categories Distribution'
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }

    renderRiskChart() {
        const ctx = document.getElementById('riskChart');
        if (!ctx) return;

        // Sample data - would come from backend
        const riskData = {
            'Low Risk': 25,
            'Medium Risk': 15,
            'High Risk': 5
        };

        const labels = Object.keys(riskData);
        const values = Object.values(riskData);
        const colors = [
            'rgba(34, 197, 94, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(239, 68, 68, 0.8)'
        ];

        this.charts.risk = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of Items',
                    data: values,
                    backgroundColor: colors,
                    borderColor: colors.map(color => color.replace('0.8', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Risk Distribution'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }

    renderStockChart() {
        const ctx = document.getElementById('stockChart');
        if (!ctx) return;

        // Sample data - would come from backend
        const stockData = {
            'Low Stock': 8,
            'Optimal Stock': 30,
            'Overstocked': 7
        };

        const labels = Object.keys(stockData);
        const values = Object.values(stockData);
        const colors = [
            'rgba(239, 68, 68, 0.8)',
            'rgba(34, 197, 94, 0.8)',
            'rgba(245, 158, 11, 0.8)'
        ];

        this.charts.stock = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of Items',
                    data: values,
                    backgroundColor: colors,
                    borderColor: colors.map(color => color.replace('0.8', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Stock Level Distribution'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }

    renderRecommendations() {
        const container = document.getElementById('recommendations-list');
        if (!container || !this.data.recommendations) return;

        container.innerHTML = '';

        if (this.data.recommendations.length === 0) {
            container.innerHTML = '<p class="no-recommendations">No recommendations at this time.</p>';
            return;
        }

        this.data.recommendations.forEach(rec => {
            const recElement = this.createRecommendationElement(rec);
            container.appendChild(recElement);
        });
    }

    createRecommendationElement(rec) {
        const div = document.createElement('div');
        div.className = `recommendation recommendation-${rec.type}`;
        
        div.innerHTML = `
            <div class="rec-content">
                <div class="rec-main">
                    <strong>${rec.message}</strong>
                    <div class="rec-meta">
                        <span class="rec-priority priority-${rec.priority}">${rec.priority.toUpperCase()}</span>
                        ${rec.suggested_quantity ? `<span class="rec-quantity">Suggested: ${rec.suggested_quantity}</span>` : ''}
                    </div>
                </div>
                <div class="rec-actions">
                    <button class="btn btn-sm btn-primary" onclick="handleRecommendation('${rec.action}', ${rec.medicine_id})">
                        ${this.getActionText(rec.action)}
                    </button>
                </div>
            </div>
        `;

        return div;
    }

    getActionText(action) {
        const actionTexts = {
            'reorder_immediately': 'Reorder Now',
            'reorder_soon': 'Plan Reorder',
            'promote_usage': 'Promote Usage',
            'reduce_stock': 'Reduce Stock',
            'review_urgently': 'Review'
        };
        return actionTexts[action] || 'Take Action';
    }

    refreshChart(chartType) {
        if (this.charts[chartType]) {
            this.charts[chartType].destroy();
        }
        this.loadAnalyticsData();
    }

    updateDateRange(days) {
        this.loadAnalyticsData();
    }
}

// Consumption tracking functionality
class ConsumptionTracker {
    constructor() {
        this.setupConsumptionModal();
    }

    setupConsumptionModal() {
        // Modal functionality is already in dashboard.html
        // This class can be extended for additional consumption tracking features
    }

    async recordConsumption(medicineId, quantity, notes = '') {
        try {
            const response = await fetch('/analytics/api/consumption', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    medicine_id: medicineId,
                    quantity: quantity,
                    notes: notes
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.showNotification('Consumption recorded successfully', 'success');
                return result;
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('Error recording consumption:', error);
            this.showNotification('Error recording consumption: ' + error.message, 'error');
            throw error;
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '12px 20px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '600',
            zIndex: '10000',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease'
        });

        // Set background color based on type
        const colors = {
            success: '#22c55e',
            error: '#ef4444',
            warning: '#f59e0b',
            info: '#3b82f6'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        // Add to page
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
}

// Data export functionality
class DataExporter {
    static async exportToCSV(format = 'all') {
        try {
            const response = await fetch('/reports/api/export');
            const data = await response.json();
            
            // Filter data based on format
            let filteredData = data;
            if (format === 'medicines') {
                filteredData = data.filter(item => item.Name);
            }
            
            const csv = this.convertToCSV(filteredData);
            this.downloadFile(csv, `inventory_export_${format}_${new Date().toISOString().split('T')[0]}.csv`, 'text/csv');
        } catch (error) {
            console.error('Error exporting data:', error);
            alert('Error exporting data');
        }
    }

    static async exportToJSON() {
        try {
            const response = await fetch('/reports/api/export');
            const data = await response.json();
            
            const jsonData = {
                data: data,
                exported_at: new Date().toISOString(),
                version: '1.0'
            };
            
            const json = JSON.stringify(jsonData, null, 2);
            this.downloadFile(json, `inventory_export_${new Date().toISOString().split('T')[0]}.json`, 'application/json');
        } catch (error) {
            console.error('Error exporting data:', error);
            alert('Error exporting data');
        }
    }

    static convertToCSV(data) {
        if (data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => headers.map(header => `"${row[header]}"`).join(','))
        ].join('\n');
        
        return csvContent;
    }

    static downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }
}

// Global functions for template compatibility
function handleRecommendation(action, medicineId) {
    switch (action) {
        case 'reorder_immediately':
        case 'reorder_soon':
            // Navigate to add medicine page with pre-filled data
            window.location.href = '/add?action=reorder&medicine_id=' + medicineId;
            break;
        case 'promote_usage':
            // Show consumption modal
            consumeMedicine(medicineId);
            break;
        case 'reduce_stock':
            // Navigate to medicines page with filter
            window.location.href = '/medicines?filter=overstocked';
            break;
        case 'review_urgently':
            // Navigate to medicine details
            window.location.href = '/medicines#medicine-' + medicineId;
            break;
        default:
            console.log('Unknown action:', action);
    }
}

// Initialize analytics when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize analytics dashboard if on analytics page
    if (document.getElementById('consumptionChart')) {
        window.analyticsDashboard = new AnalyticsDashboard();
    }
    
    // Initialize consumption tracker
    window.consumptionTracker = new ConsumptionTracker();
    
    // Setup global export functions
    window.exportToCSV = DataExporter.exportToCSV;
    window.exportToJSON = DataExporter.exportToJSON;
});
