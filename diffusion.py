import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def gather_timestep_coeff(schedule, timesteps, x_shape):
    """
    In diffusion, we keep 1D schedules (e.g., $\sqrt{\bar\alpha_t}$) indexed by
    timestep. This helper picks the coefficient for each item in the batch and
    reshapes it so it can multiply an image/feature tensor of shape `x_shape`.

    To use this function, consider the example
        alpha_t * x
    To compute this in code, we would write
        gather_timestep_coeff(alpha, t, x.shape) * x

    Args:
        schedule: 1D tensor containing values for each timestep.
        timesteps: 1D tensor containing a batch of time indices.
        x_shape: The reference shape.
    Returns:
        The timestep coefficients, reshaped for broadcasting.
    """
    b, *_ = timesteps.shape
    out = schedule.gather(-1, timesteps)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Passes the input timesteps through the cosine schedule for the diffusion process
    Args:
        timesteps: Number of timesteps int. 
        s: The strength of the schedule.
    Returns:
        1D tensor of length timesteps with computed alphas indexed from t = 0 to t = T-1.
    """
    # Define alpha_bar for t in [0, T], then compute alpha_t for t = 0..T-1.
    steps = timesteps
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # Pre-compute the coefficients for the diffusion process using scheduler
        # Hint: helpful to define both alpha_bar and alpha_bar_t_minus_1
        # Hint: at t = 0, alpha_bar = 1
        # Hint: what does PyTorch register_buffer do?

        # alpha_t for t = 0 .. T-1  (cosine schedule, shape [T])
        alphas = cosine_schedule(self.num_timesteps)

        # alpha_bar_t = alpha_0 * alpha_1 * ... * alpha_t  (running product)
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # alpha_bar_{t-1}: shift right by one and put 1.0 at the front
        # (at the first step there is no earlier step, so alpha_bar = 1)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # register_buffer = store a tensor inside the module so that it moves
        # with .to(device) and is saved in state_dict, but it is NOT trained.
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Coefficients for the forward process: x_t = sqrt(abar) x0 + sqrt(1-abar) eps
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # Coefficients to estimate x0 from x_t and predicted noise (Eq. 19):
        # x0 = (1/sqrt(abar)) * x_t - sqrt(1/abar - 1) * eps
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

        # Posterior q(x_{t-1} | x_t, x0) (Eq. 18)
        # mean = coef1 * x_t + coef2 * x0_hat
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas_cumprod_prev) * (1.0 - alphas) / (1.0 - alphas_cumprod),
        )
        # variance = (1 - abar_{t-1}) / (1 - abar_t) * (1 - alpha_t)
        self.register_buffer(
            "posterior_variance",
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * (1.0 - alphas),
        )
    
    ##############################################
    # FORWARD DIFFUSION    
    ##############################################

    def q_sample(self, x_0, t, noise):
        """
        Applies alpha interpolation between x_0 and noise to simulate sampling
        x_t from the noise distribution.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled images.
        """
        ##### TODO: Implement the q_sample function #######
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = (
            gather_timestep_coeff(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + gather_timestep_coeff(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        return x_t
        
    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss

        # 1) make the noisy image x_t from x_0 (forward process)
        x_t = self.q_sample(x_0, t, noise)
        # 2) the U-Net tries to predict the noise that we added
        predicted_noise = self.model(x_t, t)
        # 3) L1 loss between true noise and predicted noise (mean over all elements)
        loss = F.l1_loss(predicted_noise, noise)
        return loss
    
    def forward(self, x_0, noise):
        """
        Acts as a wrapper for p_losses.
        Args:
            x_0: The initial images.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ##### TODO: Implement the forward function ########
        # Step 3 of Algorithm 1: t ~ Uniform({0, ..., T-1}), one t for each image
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x_0, t, noise)

    ############################################################
    # BACKWARD DIFFUSION
    ############################################################

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Computes the (t_index)th sample from the (t_index + 1)th sample using
        the reverse diffusion process.
        Args:
            x: The sampled image at timestep t_index + 1.
            t: 1D tensor of the index of the time step.
            t_index: Scalar of the index of the time step.
        Returns:
            The sampled image at timestep t_index.
        """
        ## TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use gather_timestep_coeff function to get the coefficients at time t
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Begin code here
        # (a) predicted noise from the U-Net
        predicted_noise = self.model(x, t)

        # (b) estimate the clean image x0 (Eq. 19) and clamp to the valid range [-1, 1]
        x_0_hat = (
            gather_timestep_coeff(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - gather_timestep_coeff(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * predicted_noise
        )
        x_0_hat = torch.clamp(x_0_hat, -1.0, 1.0)

        # (c) posterior mean of x_{t-1}  (Eq. 18)
        posterior_mean = (
            gather_timestep_coeff(self.posterior_mean_coef1, t, x.shape) * x
            + gather_timestep_coeff(self.posterior_mean_coef2, t, x.shape) * x_0_hat
        )

        if t_index == 0:
            # last step: no extra noise (z = 0)
            return posterior_mean
        else:
            # (d) add random noise scaled by the posterior standard deviation
            posterior_variance = gather_timestep_coeff(self.posterior_variance, t, x.shape)
            noise = self.noise_like(x.shape, x.device)
            return posterior_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Passes noise through the entire reverse diffusion process to generate
        final image samples.
        Args:
            img: The initial noise that is randomly sampled from the noise distribution.
        Returns:
            The sampled images.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generated image to valid pixel range
        # Hint: to get time index, you can use torch.full()
        for t_index in tqdm(reversed(range(self.num_timesteps)), desc="sampling", total=self.num_timesteps, leave=False):
            t = torch.full((b,), t_index, device=img.device, dtype=torch.long)
            img = self.p_sample(img, t, t_index)
        # clamp to [-1, 1] and map to [0, 1] so it can be saved as a picture
        img = torch.clamp(img, -1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Wrapper function for p_sample_loop.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        #### TODO: Implement the sample function ####
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = self.noise_like(shape, self.alphas.device)
        return self.p_sample_loop(img)
        # ####################################################
