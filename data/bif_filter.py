class BayesianNetworkFilter:
    def __init__(self, input_dir='./raw/all', output_dir='./filtered_networks'):
        from pathlib import Path


        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = {
            'total_networks': 0,
            'under_50_nodes': 0,
            'complete_cpts': 0,
            'final_filtered': 0,
            'failed_to_load': 0
        }
        
        self.network_details = []
        
    def load_network(self, bif_file):
        from pgmpy.readwrite import BIFReader

        try:
            reader = BIFReader(str(bif_file))
            model = reader.get_model()
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load {bif_file.name}: {e}")
            self.stats['failed_to_load'] += 1
            return None
        
    def has_50_or_fewer_nodes(self, model):
        return len(model.nodes()) <= 50
    
    
    def has_complete_cpts(self,model):
        import numpy as np

        if not model.check_model():
            return False

        for cpd in model.get_cpds():
            vals = np.asarray(cpd.values)

            var_card = vals.shape[0]         

            flat = vals.reshape(var_card, -1)


            parents = cpd.variables[1:]
            if parents:
                parent_cards = [cpd.get_cardinality([p])[p] for p in parents]
                expected = int(np.prod(parent_cards))
                if flat.shape[1] != expected:
                    return False

        return True
    
    def get_network_info(self, model, network_name):

        try:
            info = {
                'name': network_name,
                'n_nodes': len(model.nodes()),
                'n_edges': len(model.edges()),
                'n_cpds': len(model.get_cpds()),
                'nodes': list(model.nodes()),
                'max_cardinality': max([model.get_cardinality(node) for node in model.nodes()]),
                'under_50_nodes': self.has_50_or_fewer_nodes(model),
                'complete_cpts': self.has_complete_cpts(model)
            }
            return info
        
        except Exception as e:
            print(f"[ERROR] Could not extract network info: {e}")
            return {
                'name': network_name,
                'error': str(e)
                }
    
    def filter_networks(self, verbose=True):
        import shutil

        bif_files = list(self.input_dir.glob('*.bif'))
        self.stats['total_networks'] = len(bif_files)
        
        if verbose:
            print(f"[INFO] Found {len(bif_files)} BIF files to filter")
            print("=" * 60)
        
        filtered_networks = []
        
        for bif_file in bif_files:

            network_name = bif_file.stem
            
            if verbose:
                print(f"\nProcessing: {network_name}")
            
            model = self.load_network(bif_file)

            if model is None:
                continue
            
            info = self.get_network_info(model, network_name)
            self.network_details.append(info)
            
    
            passed_filters = True

            if not self.has_50_or_fewer_nodes(model):
                if verbose:
                    print(f"[ERROR] Failed: Too many nodes ({info.get('num_nodes', 'Unknown')} > 50)")
                passed_filters = False
            else:
                self.stats['under_50_nodes'] += 1
                if verbose:
                    print(f"[INFO] Passed: Node count acceptable ({info.get('num_nodes', 'Unknown')} <= 50)")
            

            if not self.has_complete_cpts(model):
                if verbose:
                    print("[ERROR] Failed: Incomplete CPT information")
                passed_filters = False
            else:
                self.stats['complete_cpts'] += 1
                if verbose:
                    print("[INFO] Passed: Complete CPT information")
            

            if passed_filters:
                self.stats['final_filtered'] += 1
                filtered_networks.append(network_name)
                
                # Copy file to output directory
                output_file = self.output_dir / bif_file.name
                shutil.copy2(bif_file, output_file)
                
                if verbose:
                    print(f"[INFO] ACCEPTED: Copied to {output_file}")
            else:
                if verbose:
                    print(f"[INFO] REJECTED")
        
        return filtered_networks
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("FILTERING SUMMARY")
        print("=" * 60)
        
        print(f"Total networks processed: {self.stats['total_networks']}")
        print(f"Failed to load: {self.stats['failed_to_load']}")
        print(f"Successfully loaded: {self.stats['total_networks'] - self.stats['failed_to_load']}")

        print("\nFilter Results:")
        print(f"<=50 nodes: {self.stats['under_50_nodes']}")
        print(f"Complete CPTs: {self.stats['complete_cpts']}\n")
        
        print(f"FINAL FILTERED NETWORKS: {self.stats['final_filtered']}")
        
        if self.stats['total_networks'] > 0:
            percentage = (self.stats['final_filtered'] / self.stats['total_networks']) * 100
            print(f"Success rate: {percentage:.1f}%")
    
    

def main():
    filter_tool = BayesianNetworkFilter(
        input_dir='./raw/all',
        output_dir='./temp'
    )
    
    filtered_networks = filter_tool.filter_networks(verbose=True)
    
    filter_tool.print_summary()

    print(f"\n[INFO] Filtered networks saved to: {filter_tool.output_dir}")
    print(f"[INFO] Network names: {filtered_networks}")


if __name__ == "__main__":
    main()