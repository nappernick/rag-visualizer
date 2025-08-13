import React, { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
// @ts-ignore
import fcose from 'cytoscape-fcose';
import type { Entity, Relationship } from '../../types';

// Register the fcose layout
cytoscape.use(fcose);

interface GraphViewerProps {
  entities: Entity[];
  relationships: Relationship[];
  onNodeSelect?: (entityId: string) => void;
  selectedEntityId?: string;
}

export const GraphViewer: React.FC<GraphViewerProps> = ({
  entities,
  relationships,
  onNodeSelect,
  selectedEntityId,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current || entities.length === 0) return;

    // Prepare graph data
    const nodes = entities.map(entity => ({
      data: {
        id: entity.id,
        label: entity.name,
        type: entity.entity_type,
        frequency: entity.frequency,
      },
    }));

    const edges = relationships.map(rel => ({
      data: {
        id: `${rel.source_entity_id}-${rel.target_entity_id}`,
        source: rel.source_entity_id,
        target: rel.target_entity_id,
        label: rel.relationship_type,
        weight: rel.weight,
      },
    }));

    // Initialize Cytoscape
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: [...nodes, ...edges],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele: any) => {
              const type = ele.data('type');
              const colors: Record<string, string> = {
                person: '#3b82f6',
                organization: '#10b981',
                location: '#f59e0b',
                product: '#8b5cf6',
                model: '#ef4444',
                framework: '#ec4899',
                database: '#06b6d4',
                concept: '#6b7280',
              };
              return colors[type] || '#9ca3af';
            },
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '12px',
            'width': (ele: any) => Math.min(20 + ele.data('frequency') * 5, 60),
            'height': (ele: any) => Math.min(20 + ele.data('frequency') * 5, 60),
            'border-width': (ele: any) => ele.id() === selectedEntityId ? 3 : 0,
            'border-color': '#000',
          },
        },
        {
          selector: 'edge',
          style: {
            'width': (ele: any) => Math.min(1 + ele.data('weight'), 5),
            'line-color': '#9ca3af',
            'target-arrow-color': '#9ca3af',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '10px',
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
          },
        },
        {
          selector: ':selected',
          style: {
            'background-color': '#fbbf24',
            'border-width': 3,
            'border-color': '#000',
          },
        },
      ],
      layout: {
        name: 'fcose',
        animate: true,
        animationDuration: 1000,
        fit: true,
        padding: 50,
        nodeRepulsion: 4500,
        idealEdgeLength: 100,
        edgeElasticity: 0.45,
        nestingFactor: 0.1,
        numIter: 2500,
        tile: true,
        tilingPaddingVertical: 10,
        tilingPaddingHorizontal: 10,
        gravity: 0.25,
        gravityRange: 3.8,
      } as any,
    });

    // Add event listeners
    cyRef.current.on('tap', 'node', (evt) => {
      const node = evt.target;
      if (onNodeSelect) {
        onNodeSelect(node.id());
      }
    });

    // Cleanup
    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, [entities, relationships, selectedEntityId]);

  const handleZoomIn = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 1.2);
    }
  };

  const handleZoomOut = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 0.8);
    }
  };

  const handleFit = () => {
    if (cyRef.current) {
      cyRef.current.fit();
    }
  };

  const handleLayout = (layoutName: string) => {
    if (cyRef.current) {
      const layout = cyRef.current.layout({ name: layoutName } as any);
      layout.run();
    }
  };

  return (
    <div className="graph-viewer relative h-full">
      <div className="absolute top-4 right-4 z-10 bg-white rounded-lg shadow-lg p-2">
        <div className="flex flex-col gap-2">
          <button
            onClick={handleZoomIn}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
            title="Zoom In"
          >
            +
          </button>
          <button
            onClick={handleZoomOut}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
            title="Zoom Out"
          >
            -
          </button>
          <button
            onClick={handleFit}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
            title="Fit to Screen"
          >
            ⊡
          </button>
        </div>
      </div>
      
      <div className="absolute top-4 left-4 z-10 bg-white rounded-lg shadow-lg p-2">
        <div className="flex gap-2">
          <button
            onClick={() => handleLayout('fcose')}
            className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 text-sm"
          >
            Force
          </button>
          <button
            onClick={() => handleLayout('circle')}
            className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 text-sm"
          >
            Circle
          </button>
          <button
            onClick={() => handleLayout('grid')}
            className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 text-sm"
          >
            Grid
          </button>
          <button
            onClick={() => handleLayout('breadthfirst')}
            className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 text-sm"
          >
            Tree
          </button>
        </div>
      </div>

      <div className="absolute bottom-4 left-4 z-10 bg-white rounded-lg shadow-lg p-3">
        <div className="text-sm">
          <div className="font-semibold mb-2">Entity Types</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>Person</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>Organization</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <span>Location</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
              <span>Product</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span>Model</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-pink-500 rounded-full"></div>
              <span>Framework</span>
            </div>
          </div>
        </div>
      </div>

      <div 
        ref={containerRef} 
        className="w-full h-full bg-gray-50"
        style={{ minHeight: '600px' }}
      />
    </div>
  );
};