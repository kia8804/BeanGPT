import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line, Bar, Scatter, Pie } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ChartComponent = ({ chartData, darkMode }) => {
  if (!chartData || chartData.error) {
    return (
      <div className={`p-4 rounded-lg border ${darkMode ? 'bg-slate-800 border-slate-700 text-slate-300' : 'bg-gray-50 border-gray-200 text-gray-600'}`}>
        <p className="text-sm">{chartData?.error || 'No chart data available'}</p>
      </div>
    );
  }

  // Validate chart data structure
  if (!chartData.type || !chartData.data) {
    return (
      <div className={`p-4 rounded-lg border ${darkMode ? 'bg-slate-800 border-slate-700 text-slate-300' : 'bg-gray-50 border-gray-200 text-gray-600'}`}>
        <p className="text-sm">Invalid chart data structure</p>
        <pre className="text-xs mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded">
          {JSON.stringify(chartData, null, 2)}
        </pre>
      </div>
    );
  }

  // Configure chart options with dark mode support
  const getChartOptions = (baseOptions, chartType) => {
    const commonOptions = {
      ...baseOptions,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        ...baseOptions.plugins,
        legend: {
          ...baseOptions.plugins?.legend,
          display: true,
          position: chartType === 'scatter' ? 'right' : 'top',
          labels: {
            color: darkMode ? '#e2e8f0' : '#374151',
            font: {
              size: 12
            },
            usePointStyle: chartType === 'scatter',
            pointStyle: 'circle'
          }
        },
        title: {
          ...baseOptions.plugins?.title,
          display: true,
          text: chartData.title,
          color: darkMode ? '#f1f5f9' : '#1f2937',
          font: {
            size: 16,
            weight: 'bold'
          }
        },
        tooltip: {
          ...baseOptions.plugins?.tooltip,
          callbacks: {
            title: function(context) {
              if (chartType === 'scatter' && context[0]?.dataset?.label) {
                return context[0].dataset.label;
              }
              return baseOptions.plugins?.tooltip?.callbacks?.title || 'Data Point';
            },
            label: function(context) {
              if (chartType === 'scatter') {
                const label = context.dataset.label || '';
                const x = context.parsed.x;
                const y = context.parsed.y;
                return `${label}: (${x}, ${y})`;
              }
              return context.dataset.label + ': ' + context.formattedValue;
            }
          }
        }
      }
    };

    // Pie charts don't use scales
    if (chartType === 'pie') {
      return commonOptions;
    }

    // Add scales for other chart types
    return {
      ...commonOptions,
      scales: {
        ...baseOptions.scales,
        x: {
          ...baseOptions.scales?.x,
          ticks: {
            color: darkMode ? '#94a3b8' : '#6b7280'
          },
          grid: {
            color: darkMode ? '#334155' : '#e5e7eb'
          },
          title: {
            ...baseOptions.scales?.x?.title,
            color: darkMode ? '#cbd5e1' : '#4b5563'
          }
        },
        y: {
          ...baseOptions.scales?.y,
          ticks: {
            color: darkMode ? '#94a3b8' : '#6b7280'
          },
          grid: {
            color: darkMode ? '#334155' : '#e5e7eb'
          },
          title: {
            ...baseOptions.scales?.y?.title,
            color: darkMode ? '#cbd5e1' : '#4b5563'
          }
        }
      }
    };
  };

  const renderChart = () => {
    const options = getChartOptions(chartData.options || {}, chartData.type);

    switch (chartData.type) {
      case 'line':
        return <Line data={chartData.data} options={options} />;
      
      case 'bar':
        return <Bar data={chartData.data} options={options} />;
      
      case 'horizontalBar':
        // Handle horizontalBar by converting to bar with indexAxis: 'y'
        const horizontalOptions = {
          ...options,
          indexAxis: 'y',
        };
        return <Bar data={chartData.data} options={horizontalOptions} />;
      
      case 'scatter':
        return <Scatter data={chartData.data} options={options} />;
      
      case 'pie':
        return <Pie data={chartData.data} options={options} />;
      
      default:
        return (
          <div className={`p-4 text-center ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>
            <p>Unsupported chart type: {chartData.type}</p>
            <p className="text-xs mt-2">Available types: line, bar, horizontalBar, scatter, pie</p>
            <pre className="text-xs mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded">
              {JSON.stringify(chartData, null, 2)}
            </pre>
          </div>
        );
    }
  };

  return (
    <div className={`p-4 rounded-lg border ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'}`}>
      <div style={{ height: '400px' }}>
        {renderChart()}
      </div>
    </div>
  );
};

export default ChartComponent; 