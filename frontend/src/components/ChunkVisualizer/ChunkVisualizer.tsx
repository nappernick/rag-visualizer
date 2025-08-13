import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { Chunk } from '../../types';

interface ChunkVisualizerProps {
  chunks: Chunk[];
  selectedChunkId?: string;
  onChunkSelect?: (chunkId: string) => void;
}

export const ChunkVisualizer: React.FC<ChunkVisualizerProps> = ({
  chunks,
  selectedChunkId,
  onChunkSelect,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [viewMode, setViewMode] = useState<'hierarchy' | 'linear' | 'grid'>('hierarchy');

  useEffect(() => {
    if (!svgRef.current || chunks.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 1200;
    const height = viewMode === 'hierarchy' ? Math.max(600, chunks.length * 30) : 600;
    const margin = { top: 40, right: 40, bottom: 40, left: 40 };

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    if (viewMode === 'hierarchy') {
      renderHierarchy(svg, chunks, width, height, margin);
    } else if (viewMode === 'linear') {
      renderLinear(svg, chunks, width, height, margin);
    } else {
      renderGrid(svg, chunks, width, height, margin);
    }
  }, [chunks, viewMode, selectedChunkId]);

  const renderHierarchy = (
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    chunks: Chunk[],
    width: number,
    height: number,
    margin: { top: number; right: number; bottom: number; left: number }
  ) => {
    // Build hierarchy data
    const root = buildHierarchyData(chunks);
    console.log('Hierarchy root:', root);
    console.log('Number of chunks:', chunks.length);
    console.log('Root children:', root.children.length);
    
    const treeLayout = d3.tree()
      .size([width - margin.left - margin.right, height - margin.top - margin.bottom - 40])
      .separation((a, b) => a.parent === b.parent ? 1 : 1.5);

    const hierarchyRoot = d3.hierarchy(root);
    const treeData = treeLayout(hierarchyRoot);
    console.log('Tree descendants:', treeData.descendants().length);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Draw links
    g.selectAll('.link')
      .data(treeData.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkVertical()
        .x((d: any) => d.x)
        .y((d: any) => d.y) as any)
      .style('fill', 'none')
      .style('stroke', '#ccc')
      .style('stroke-width', 2);

    // Draw nodes
    const nodes = g.selectAll('.node')
      .data(treeData.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (d: any) => `translate(${d.x}, ${d.y})`);

    nodes.append('circle')
      .attr('r', (d: any) => {
        const chunk = d.data.chunk;
        if (!chunk) return 12; // Root node
        return chunk.chunk_type === 'summary' ? 15 : 10;
      })
      .style('fill', (d: any) => {
        const chunk = d.data.chunk;
        if (!chunk) return '#999';
        const colors: Record<string, string> = {
          summary: '#8b5cf6',
          section: '#3b82f6',
          hierarchical: '#10b981',
          standard: '#f59e0b',
          code: '#ef4444',
          table: '#ec4899',
        };
        return colors[chunk.chunk_type] || '#999';
      })
      .style('stroke', (d: any) => {
        const chunk = d.data.chunk;
        return chunk && chunk.id === selectedChunkId ? '#000' : '#fff';
      })
      .style('stroke-width', (d: any) => {
        const chunk = d.data.chunk;
        return chunk && chunk.id === selectedChunkId ? 3 : 2;
      })
      .style('cursor', 'pointer')
      .on('click', (_event, d: any) => {
        if (d.data.chunk && onChunkSelect) {
          onChunkSelect(d.data.chunk.id);
        }
      });

    // Add labels
    nodes.append('text')
      .attr('dy', -18)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-weight', (d: any) => d.data.chunk ? 'normal' : 'bold')
      .text((d: any) => d.data.name);

    // Add tooltips
    nodes.append('title')
      .text((d: any) => {
        const chunk = d.data.chunk;
        if (!chunk) return d.data.name;
        return `Type: ${chunk.chunk_type}\nTokens: ${chunk.tokens}\nIndex: ${chunk.chunk_index}`;
      });
  };

  const renderLinear = (
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    chunks: Chunk[],
    width: number,
    height: number,
    margin: { top: number; right: number; bottom: number; left: number }
  ) => {
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    const xScale = d3.scaleLinear()
      .domain([0, chunks.length - 1])
      .range([0, width - margin.left - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(chunks, c => c.tokens) || 100])
      .range([height - margin.top - margin.bottom, 0]);

    // Draw bars
    g.selectAll('.chunk-bar')
      .data(chunks)
      .enter()
      .append('rect')
      .attr('class', 'chunk-bar')
      .attr('x', (d, i) => xScale(i))
      .attr('y', d => yScale(d.tokens))
      .attr('width', (width - margin.left - margin.right) / chunks.length - 2)
      .attr('height', d => height - margin.top - margin.bottom - yScale(d.tokens))
      .style('fill', d => {
        const colors: Record<string, string> = {
          summary: '#8b5cf6',
          section: '#3b82f6',
          hierarchical: '#10b981',
          standard: '#f59e0b',
          code: '#ef4444',
          table: '#ec4899',
        };
        return colors[d.chunk_type] || '#999';
      })
      .style('stroke', d => d.id === selectedChunkId ? '#000' : 'none')
      .style('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (_event, d) => {
        if (onChunkSelect) {
          onChunkSelect(d.id);
        }
      });

    // Add x-axis
    g.append('g')
      .attr('transform', `translate(0, ${height - margin.top - margin.bottom})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `Chunk ${d}`));

    // Add y-axis
    g.append('g')
      .call(d3.axisLeft(yScale));

    // Add y-axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -margin.left)
      .attr('x', -(height - margin.top - margin.bottom) / 2)
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text('Tokens');
  };

  const renderGrid = (
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    chunks: Chunk[],
    width: number,
    height: number,
    margin: { top: number; right: number; bottom: number; left: number }
  ) => {
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    const cols = Math.ceil(Math.sqrt(chunks.length));
    const rows = Math.ceil(chunks.length / cols);

    const cellWidth = (width - margin.left - margin.right) / cols;
    const cellHeight = (height - margin.top - margin.bottom) / rows;

    g.selectAll('.chunk-cell')
      .data(chunks)
      .enter()
      .append('g')
      .attr('class', 'chunk-cell')
      .attr('transform', (d, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        return `translate(${col * cellWidth}, ${row * cellHeight})`;
      })
      .each(function(d) {
        const cell = d3.select(this);

        // Draw cell rectangle
        cell.append('rect')
          .attr('width', cellWidth - 4)
          .attr('height', cellHeight - 4)
          .attr('x', 2)
          .attr('y', 2)
          .style('fill', () => {
            const colors: Record<string, string> = {
              summary: '#8b5cf6',
              section: '#3b82f6',
              hierarchical: '#10b981',
              standard: '#f59e0b',
              code: '#ef4444',
              table: '#ec4899',
            };
            return colors[d.chunk_type] || '#999';
          })
          .style('opacity', 0.7)
          .style('stroke', d.id === selectedChunkId ? '#000' : '#fff')
          .style('stroke-width', d.id === selectedChunkId ? 3 : 1)
          .style('cursor', 'pointer')
          .on('click', () => {
            if (onChunkSelect) {
              onChunkSelect(d.id);
            }
          });

        // Add text
        cell.append('text')
          .attr('x', cellWidth / 2)
          .attr('y', cellHeight / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('font-size', '10px')
          .style('fill', '#fff')
          .text(`#${d.chunk_index}`);

        // Add tooltip
        cell.append('title')
          .text(`Type: ${d.chunk_type}\nTokens: ${d.tokens}\nIndex: ${d.chunk_index}`);
      });
  };

  const buildHierarchyData = (chunks: Chunk[]) => {
    const root: any = {
      name: 'Document',
      children: [],
      chunk: null,
    };

    const nodeMap = new Map();
    nodeMap.set('root', root);

    // First pass: create all nodes
    chunks.forEach(chunk => {
      const node = {
        name: `Chunk ${chunk.chunk_index}`,
        children: [],
        chunk: chunk,
      };
      nodeMap.set(chunk.id, node);
    });

    // Second pass: build hierarchy
    chunks.forEach(chunk => {
      const node = nodeMap.get(chunk.id);
      if (chunk.parent_id && nodeMap.has(chunk.parent_id)) {
        const parent = nodeMap.get(chunk.parent_id);
        parent.children.push(node);
      } else if (!chunk.parent_id) {
        root.children.push(node);
      }
    });

    // If no hierarchy, show flat structure
    if (root.children.length === 0) {
      console.log('No hierarchical structure found, creating flat structure');
      chunks.forEach(chunk => {
        const node = nodeMap.get(chunk.id);
        if (node) {
          root.children.push(node);
        }
      });
    }

    return root;
  };

  return (
    <div className="chunk-visualizer">
      <div className="controls mb-4">
        <button
          className={`px-3 py-1 mr-2 rounded ${viewMode === 'hierarchy' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
          onClick={() => setViewMode('hierarchy')}
        >
          Hierarchy
        </button>
        <button
          className={`px-3 py-1 mr-2 rounded ${viewMode === 'linear' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
          onClick={() => setViewMode('linear')}
        >
          Linear
        </button>
        <button
          className={`px-3 py-1 rounded ${viewMode === 'grid' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
          onClick={() => setViewMode('grid')}
        >
          Grid
        </button>
      </div>
      <svg ref={svgRef} className="w-full h-full border border-gray-300 rounded"></svg>
    </div>
  );
};