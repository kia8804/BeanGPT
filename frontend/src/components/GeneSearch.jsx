import { useState, useEffect, useRef } from 'react';
import { FaSearch, FaSpinner } from 'react-icons/fa';

function GeneSearch({ apiKey }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [loadingStep, setLoadingStep] = useState('');
  const searchTimeoutRef = useRef(null);

  const handleSearch = async (searchQuery) => {
    try {
      if (!apiKey) {
        setError('Please enter your API key to use the gene search feature.');
        return;
      }

      setIsLoading(true);
      setError(null);
      setLoadingStep('Searching databases...');
      
      // Simulate search steps for better UX
      const steps = [
        'Searching NCBI database...',
        'Searching UniProt database...',
        'Analyzing results...',
        'Generating AI description...',
        'Finalizing results...'
      ];

      let currentStep = 0;
      const stepInterval = setInterval(() => {
        if (currentStep < steps.length - 1) {
          currentStep++;
          setLoadingStep(steps[currentStep]);
        }
      }, 800);
      
      const response = await fetch('/api/gene-search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query: searchQuery,
          api_key: apiKey 
        }),
      });

      clearInterval(stepInterval);

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError('Failed to search genes. Please try again.');
      console.error('Search error:', err);
    } finally {
      setIsLoading(false);
      setLoadingStep('');
    }
  };

  const handleSearchSubmit = () => {
    if (query.trim().length >= 2) {
      handleSearch(query.trim());
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSearchSubmit();
    }
  };

  useEffect(() => {
    // Clear results when query is too short
    if (query.trim().length < 2) {
      setResults(null);
      setError(null);
      setLoadingStep('');
    }
  }, [query]);

  const renderGeneResult = (gene) => (
    <div key={gene.id} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-4">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-semibold text-blue-600 dark:text-blue-400">
            {gene.name}
          </h3>
          {gene.aliases && gene.aliases.length > 0 && (
            <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Also known as: {gene.aliases.join(', ')}
            </div>
          )}
        </div>
        {gene.source && (
          <span className={`text-sm px-3 py-1 rounded-full font-medium ${
            gene.source === 'NCBI' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
            gene.source === 'UniProt' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400' :
            'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400'
          }`}>
            {gene.source}
          </span>
        )}
      </div>

      {/* For database results */}
      {!gene.sections && (
        <>
          <div className="mt-4 prose dark:prose-invert max-w-none">
            <p className="text-gray-800 dark:text-gray-200">
              {gene.description}
            </p>
          </div>
          
          {gene.functions && gene.functions.length > 0 && (
            <div className="mt-4">
              <h4 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">Functions</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                {gene.functions.map((func, idx) => (
                  <li key={idx} className="ml-4">{func}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {/* For AI-generated results */}
      {gene.sections && (
        <div className="mt-4 space-y-6">
          {gene.sections.map((section, idx) => (
            <div key={idx} className="prose dark:prose-invert max-w-none">
              <h4 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                {section.title}
              </h4>
              <div className="text-gray-700 dark:text-gray-300">
                {section.content}
              </div>
            </div>
          ))}

          {gene.molecular_details && (
            <div className="mt-6 bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-3">
                Molecular Details
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {gene.molecular_details.domains && (
                  <div>
                    <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Domains</h5>
                    <ul className="list-disc list-inside space-y-1 text-gray-600 dark:text-gray-400 text-sm">
                      {gene.molecular_details.domains.map((domain, idx) => (
                        <li key={idx} className="ml-4">{domain}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {gene.molecular_details.interactions && (
                  <div>
                    <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Interactions</h5>
                    <ul className="list-disc list-inside space-y-1 text-gray-600 dark:text-gray-400 text-sm">
                      {gene.molecular_details.interactions.map((interaction, idx) => (
                        <li key={idx} className="ml-4">{interaction}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {gene.molecular_details.pathways && (
                  <div>
                    <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Pathways</h5>
                    <ul className="list-disc list-inside space-y-1 text-gray-600 dark:text-gray-400 text-sm">
                      {gene.molecular_details.pathways.map((pathway, idx) => (
                        <li key={idx} className="ml-4">{pathway}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {gene.technical_notes && (
            <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 className="text-sm font-medium text-blue-900 dark:text-blue-300 mb-2">Technical Notes</h4>
              <p className="text-blue-800 dark:text-blue-200 text-sm">{gene.technical_notes}</p>
            </div>
          )}
        </div>
      )}

      <div className="mt-6 flex flex-wrap gap-4">
        {gene.references && gene.references.length > 0 && (
          <div className="flex-1">
            <details className="text-sm">
              <summary className="cursor-pointer text-blue-500 hover:text-blue-600 font-medium">
                View References
              </summary>
              <div className="mt-2 space-y-2">
                {gene.references.map((ref, idx) => (
                  <a
                    key={idx}
                    href={ref.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-blue-500 hover:underline"
                  >
                    {ref.title || ref.url}
                  </a>
                ))}
              </div>
            </details>
          </div>
        )}

        {gene.suggested_bean_genes && gene.suggested_bean_genes.length > 0 && (
          <div className="flex-1">
            <details className="text-sm">
              <summary className="cursor-pointer text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 font-medium">
                Related Bean Genes
              </summary>
              <div className="mt-2">
                <ul className="list-disc list-inside space-y-1 text-emerald-700 dark:text-emerald-300">
                  {gene.suggested_bean_genes.map((gene, idx) => (
                    <li key={idx} className="ml-4">{gene}</li>
                  ))}
                </ul>
              </div>
            </details>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="w-full">
      <div className="mb-6">
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Search our comprehensive database of bean genes from NCBI and UniProt, or get AI-powered analysis for new gene queries.
        </p>
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Search for bean genes (e.g., PvNAC1, PvPRP1)... Press Enter to search"
            className="w-full p-4 pr-12 text-lg border rounded-lg shadow-sm 
                     dark:bg-gray-800 dark:border-gray-700 dark:text-white
                     focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSearchSubmit}
            disabled={isLoading || query.trim().length < 2}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 rounded-lg
                     hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <FaSpinner className="animate-spin text-blue-500" />
            ) : (
              <FaSearch className="text-blue-500 hover:text-blue-600" />
            )}
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <FaSpinner className="animate-spin text-blue-500 text-xl" />
            <div>
              <div className="text-blue-900 dark:text-blue-100 font-medium">
                Searching Gene Databases
              </div>
              <div className="text-blue-700 dark:text-blue-300 text-sm mt-1">
                {loadingStep}
              </div>
            </div>
          </div>
          
          {/* Progress bar */}
          <div className="mt-4 bg-blue-200 dark:bg-blue-800 rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-1000 ease-out"
              style={{
                width: loadingStep.includes('NCBI') ? '20%' :
                       loadingStep.includes('UniProt') ? '40%' :
                       loadingStep.includes('Analyzing') ? '60%' :
                       loadingStep.includes('AI') ? '80%' :
                       loadingStep.includes('Finalizing') ? '95%' : '10%'
              }}
            />
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg">
          {error}
        </div>
      )}

      {results && (
        <div className="mt-6 space-y-4">
          {results.length === 0 ? (
            <div className="text-center text-gray-600 dark:text-gray-400 p-8">
              <div className="text-xl mb-2">🔬</div>
              <p className="text-lg font-medium mb-2">No genes found</p>
              <p className="text-sm">
                Try searching for bean-specific genes like "PvNAC1", "PvPRP1", or check your spelling.
                {!apiKey && " Also ensure your API key is entered for AI-powered descriptions."}
              </p>
            </div>
          ) : (
            results.map(renderGeneResult)
          )}
        </div>
      )}
      
      {query.trim().length >= 2 && !isLoading && !results && !error && (
        <div className="mt-6 text-center text-gray-500 dark:text-gray-400">
          <p>Type to search for bean genes...</p>
        </div>
      )}
    </div>
  );
}

export default GeneSearch;
