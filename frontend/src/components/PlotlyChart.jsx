import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';

const PlotlyChart = ({ chartData, darkMode }) => {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const plotRef = useRef(null);
  const rootRef = useRef(null);
  const [chartKey, setChartKey] = useState(Date.now()); // Unique key for each chart

  useEffect(() => {
    // Generate new key when chartData changes to force complete re-render
    setChartKey(Date.now());
  }, [chartData]);

  // Handle responsive breakpoints
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const loadPlotly = async () => {
      // Load Plotly from CDN if not already loaded
      if (!window.Plotly) {
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-latest.min.js';
        script.onload = () => renderChart();
        document.head.appendChild(script);
      } else {
        renderChart();
      }
    };

    const renderChart = () => {
      if (!plotRef.current || !chartData || chartData.type === 'error') return;

      try {
        let plotData, layout, config;

        if (chartData.type === 'plotly' && chartData.data) {
          // Use the Plotly data directly from backend
          const figData = chartData.data;
          plotData = figData.data || [];
          layout = figData.layout || {};
          
          // Apply dark mode styling to layout
          if (darkMode) {
            layout = {
              ...layout,
              paper_bgcolor: '#1e293b',
              plot_bgcolor: '#0f172a',
              font: {
                ...layout.font,
                color: '#e2e8f0'
              }
            };

            // Update axis colors for dark mode
            if (layout.xaxis) {
              layout.xaxis = {
                ...layout.xaxis,
                gridcolor: '#374151',
                linecolor: '#6b7280',
                tickcolor: '#6b7280',
                titlefont: { color: '#e2e8f0' },
                tickfont: { color: '#e2e8f0' }
              };
            }
            if (layout.yaxis) {
              layout.yaxis = {
                ...layout.yaxis,
                gridcolor: '#374151',
                linecolor: '#6b7280',
                tickcolor: '#6b7280',
                titlefont: { color: '#e2e8f0' },
                tickfont: { color: '#e2e8f0' }
              };
            }
          }
        } else {
          // Fallback for other chart types (shouldn't happen with new system)
          console.warn('Unsupported chart data format:', chartData);
          return;
        }

        config = {
          responsive: true,
          displayModeBar: !isMobile, // Hide toolbar on mobile for cleaner look
          modeBarButtonsToRemove: isMobile 
            ? ['pan2d', 'lasso2d', 'select2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
            : ['pan2d', 'lasso2d', 'select2d'],
          displaylogo: false,
          scrollZoom: !isMobile, // Disable scroll zoom on mobile to prevent conflicts
          doubleClick: isMobile ? false : 'reset+autosize',
          toImageButtonOptions: {
            format: 'png',
            filename: 'bean_chart',
            height: isMobile ? 400 : 600,
            width: isMobile ? 600 : 1000,
            scale: 1
          }
        };

        // Mobile-specific layout adjustments
        if (isMobile && layout) {
          layout = {
            ...layout,
            margin: {
              l: 40,
              r: 20,
              t: 40,
              b: 40,
              ...layout.margin
            },
            font: {
              size: 10,
              ...layout.font
            },
            showlegend: layout.showlegend !== false,
            legend: {
              orientation: 'h',
              x: 0,
              y: -0.2,
              ...layout.legend
            }
          };

          // Adjust axis labels for mobile
          if (layout.xaxis) {
            layout.xaxis = {
              ...layout.xaxis,
              tickangle: -45,
              tickfont: { size: 9 }
            };
          }
          if (layout.yaxis) {
            layout.yaxis = {
              ...layout.yaxis,
              tickfont: { size: 9 }
            };
          }
        }

        // THOROUGH cleanup before creating new chart
        if (plotRef.current) {
          // Destroy any existing Plotly plot completely
          try {
            if (window.Plotly) {
              window.Plotly.purge(plotRef.current);
            }
          } catch (e) {
            console.warn('Error purging plot:', e);
          }
          
          // Clear all content and reset container
          plotRef.current.innerHTML = '';
          plotRef.current.style.cssText = '';
          
          // Remove any Plotly-generated classes
          plotRef.current.className = 'plotly-chart-container';
        }

        // Small delay to ensure cleanup is complete
        setTimeout(() => {
          if (plotRef.current) {
            window.Plotly.newPlot(plotRef.current, plotData, layout, config);
          }
        }, 10);

      } catch (error) {
        console.error('Error rendering Plotly chart:', error);
      }
    };

    loadPlotly();

    // Cleanup function
    return () => {
      if (plotRef.current && window.Plotly) {
        try {
          window.Plotly.purge(plotRef.current);
        } catch (e) {
          console.warn('Error during cleanup:', e);
        }
      }
    };
  }, [chartData, darkMode, chartKey]);

  if (!chartData) {
    return (
      <div className={`p-4 rounded-lg border ${darkMode ? 'bg-slate-800 border-slate-700 text-slate-300' : 'bg-gray-50 border-gray-200 text-gray-600'}`}>
        <p className="text-sm">No chart data available</p>
      </div>
    );
  }

  if (chartData.type === 'error' || chartData.error) {
    return (
      <div className={`p-4 rounded-lg border ${darkMode ? 'bg-slate-800 border-slate-700 text-slate-300' : 'bg-red-50 border-red-200 text-red-600'}`}>
        <p className="text-sm font-medium">Chart Generation Error</p>
        <p className="text-xs mt-1">{chartData.error || 'Failed to generate chart'}</p>
      </div>
    );
  }

  return (
    <div key={chartKey} className={`${isMobile ? 'p-2' : 'p-4'} rounded-lg border ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'}`}>
      <div 
        ref={plotRef} 
        key={`plot-${chartKey}`}
        style={{ 
          height: isMobile ? '300px' : '450px', 
          width: '100%',
          minHeight: isMobile ? '250px' : '400px',
          maxWidth: '100%'
        }}
        className="plotly-chart-container"
      />
      {chartData.generated_code && (
        <details className="mt-4">
          <summary className={`text-sm cursor-pointer ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>
            View Generated Code
          </summary>
          <pre className={`text-xs mt-2 p-2 rounded overflow-auto ${darkMode ? 'bg-slate-900 text-slate-300' : 'bg-gray-100 text-gray-700'}`}>
            {chartData.generated_code}
          </pre>
        </details>
      )}
    </div>
  );
};

export default PlotlyChart; 