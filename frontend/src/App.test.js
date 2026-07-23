import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the DeepRestore AI header', () => {
  render(<App />);
  const heading = screen.getByRole('heading', { name: /deeprestore ai/i });
  expect(heading).toBeInTheDocument();
});

test('renders both tool tabs', () => {
  render(<App />);
  expect(screen.getByText(/image denoiser/i)).toBeInTheDocument();
  expect(screen.getByText(/document restore/i)).toBeInTheDocument();
});
